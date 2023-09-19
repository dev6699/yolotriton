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
	ModelTypeYoloV8      ModelType = "yolov8"
	ModelTypeYoloNAS     ModelType = "yolonas"
	ModelTypeYoloNASInt8 ModelType = "yolonasint8"
)

type Model interface {
	GetConfig() YoloTritonConfig
	PreProcess(img image.Image, targetWidth uint, targetHeight uint) (*triton.InferTensorContents, error)
	PostProcess(rawOutputContents [][]byte) ([]Box, error)
}

type YoloTritonConfig struct {
	NumClasses     int
	NumObjects     int
	ModelName      string
	ModelVersion   string
	MinProbability float32
	MaxIOU         float64
	Classes        []string
}

func New(url string, model Model) (*YoloTriton, error) {
	conn, err := grpc.Dial(url, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	cfg := model.GetConfig()
	modelMetadata, err := newModelMetadata(conn, cfg.ModelName, cfg.ModelVersion)
	if err != nil {
		return nil, err
	}

	return &YoloTriton{
		conn:          conn,
		model:         model,
		cfg:           cfg,
		modelMetadata: modelMetadata,
	}, nil
}

type YoloTriton struct {
	model         Model
	cfg           YoloTritonConfig
	conn          *grpc.ClientConn
	modelMetadata *modelMetadata
}

func (y *YoloTriton) Close() error {
	return y.conn.Close()
}

func (y *YoloTriton) Infer(img image.Image) ([]Box, error) {

	inputs, err := y.model.PreProcess(img, y.modelMetadata.inputWidth(), y.modelMetadata.inputHeight())
	if err != nil {
		return nil, err
	}

	modelInferRequest := y.modelMetadata.formInferRequest(inputs)

	client := triton.NewGRPCInferenceServiceClient(y.conn)
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

type modelMetadata struct {
	modelName    string
	modelVersion string
	*triton.ModelMetadataResponse
}

func newModelMetadata(conn *grpc.ClientConn, modelName string, modelVersion string) (*modelMetadata, error) {
	client := triton.NewGRPCInferenceServiceClient(conn)
	metaResponse, err := ModelMetadataRequest(client, modelName, modelVersion)
	if err != nil {
		return nil, err
	}

	return &modelMetadata{
		modelName:             modelName,
		modelVersion:          modelVersion,
		ModelMetadataResponse: metaResponse,
	}, nil
}

func (m *modelMetadata) inputWidth() uint {
	return uint(m.Inputs[0].Shape[2])
}

func (m *modelMetadata) inputHeight() uint {
	return uint(m.Inputs[0].Shape[3])
}

func (m *modelMetadata) formInferRequest(contents *triton.InferTensorContents) *triton.ModelInferRequest {
	input := m.Inputs[0]
	if input.Shape[0] == -1 {
		input.Shape[0] = 1
	}

	outputs := make([]*triton.ModelInferRequest_InferRequestedOutputTensor, len(m.Outputs))
	for i, o := range m.Outputs {
		outputs[i] = &triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: o.Name,
		}
	}

	return &triton.ModelInferRequest{
		ModelName:    m.modelName,
		ModelVersion: m.modelVersion,
		Inputs: []*triton.ModelInferRequest_InferInputTensor{
			{
				Name:     input.Name,
				Datatype: input.Datatype,
				Shape:    input.Shape,
				Contents: contents,
			},
		},
		Outputs: outputs,
	}
}
