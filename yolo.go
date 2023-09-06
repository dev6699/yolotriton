package yolotriton

import (
	"image"
	_ "image/png"
	"sort"

	triton "github.com/dev6699/yolotriton/grpc-client"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type ModelType string

const (
	ModelTypeYoloV8  ModelType = "yolov8"
	ModelTypeYoloNAS ModelType = "yolonas"
)

type Model interface {
	GetConfig() YoloTritonConfig
	PreProcess(img image.Image, targetWidth uint, targetHeight uint) ([]float32, error)
	PostProcess(rawOutputContents [][]byte) ([]Box, error)
}

type YoloTritonConfig struct {
	BatchSize      int
	NumChannels    int
	NumObjects     int
	ModelName      string
	ModelVersion   string
	MinProbability float32
	MaxIOU         float64
}

func New(url string, model Model) (*YoloTriton, error) {
	conn, err := grpc.Dial(url, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	return &YoloTriton{
		conn:  conn,
		model: model,
		cfg:   model.GetConfig(),
	}, nil
}

type YoloTriton struct {
	cfg   YoloTritonConfig
	conn  *grpc.ClientConn
	model Model
}

func (y *YoloTriton) Close() error {
	return y.conn.Close()
}

func (y *YoloTriton) Infer(img image.Image) ([]Box, error) {

	client := triton.NewGRPCInferenceServiceClient(y.conn)

	metaResponse, err := ModelMetadataRequest(client, y.cfg.ModelName, y.cfg.ModelVersion)
	if err != nil {
		return nil, err
	}

	modelInferRequest := &triton.ModelInferRequest{
		ModelName:    y.cfg.ModelName,
		ModelVersion: y.cfg.ModelVersion,
	}

	input := metaResponse.Inputs[0]
	if input.Shape[0] == -1 {
		input.Shape[0] = 1
	}

	inputWidth := input.Shape[2]
	inputHeight := input.Shape[3]

	fp32Contents, err := y.model.PreProcess(img, uint(inputWidth), uint(inputHeight))
	if err != nil {
		return nil, err
	}

	modelInferRequest.Inputs = append(modelInferRequest.Inputs,
		&triton.ModelInferRequest_InferInputTensor{
			Name:     input.Name,
			Datatype: input.Datatype,
			Shape:    input.Shape,
			Contents: &triton.InferTensorContents{
				// Simply assume all are fp32
				Fp32Contents: fp32Contents,
			},
		},
	)

	for _, o := range metaResponse.Outputs {
		modelInferRequest.Outputs = append(modelInferRequest.Outputs,
			&triton.ModelInferRequest_InferRequestedOutputTensor{
				Name: o.Name,
			},
		)
	}

	inferResponse, err := ModelInferRequest(client, modelInferRequest)
	if err != nil {
		return nil, err
	}

	boxes, err := y.model.PostProcess(inferResponse.RawOutputContents)
	if err != nil {
		return nil, err
	}

	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Probability > boxes[j].Probability
	})

	result := []Box{}
	for len(boxes) > 0 {
		result = append(result, boxes[0])
		tmp := []Box{}
		for _, box := range boxes {
			if iou(boxes[0], box) < y.cfg.MaxIOU {
				tmp = append(tmp, box)
			}
		}
		boxes = tmp
	}

	return result, nil
}
