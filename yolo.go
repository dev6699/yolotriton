package yolotriton

import (
	"image"
	_ "image/png"

	triton "github.com/dev6699/yolotriton/grpc-client"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type YoloTritonConfig struct {
	BatchSize      int
	NumChannels    int
	NumObjects     int
	Width          int
	Height         int
	ModelName      string
	ModelVersion   string
	MinProbability float64
	MaxIOU         float64
}

func New(url string, cfg YoloTritonConfig) (*YoloTriton, error) {
	conn, err := grpc.Dial(url, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}

	return &YoloTriton{
		cfg:  cfg,
		conn: conn,
	}, nil
}

type YoloTriton struct {
	cfg  YoloTritonConfig
	conn *grpc.ClientConn
}

func (y *YoloTriton) Close() error {
	return y.conn.Close()
}

func (y *YoloTriton) Infer(img image.Image) ([]Box, error) {

	preprocessedImg := resizeImage(img, uint(y.cfg.Width), uint(y.cfg.Height))

	fp32Contents := imageToFloat32Slice(preprocessedImg)

	client := triton.NewGRPCInferenceServiceClient(y.conn)

	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		{
			Name:     "images",
			Datatype: "FP32",
			Shape:    []int64{int64(y.cfg.BatchSize), 3, int64(y.cfg.Width), int64(y.cfg.Height)},
			Contents: &triton.InferTensorContents{
				Fp32Contents: fp32Contents,
			},
		},
	}
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		{
			Name: "output0",
		},
	}
	modelInferRequest := &triton.ModelInferRequest{
		ModelName:    y.cfg.ModelName,
		ModelVersion: y.cfg.ModelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	inferResponse, err := ModelInferRequest(client, modelInferRequest)
	if err != nil {
		return nil, err
	}

	t, err := y.bytesToFloat32Slice(inferResponse.RawOutputContents[0])
	if err != nil {
		return nil, err
	}

	boxes := y.parseOutput(t, img.Bounds().Dx(), img.Bounds().Dy())
	return boxes, nil
}
