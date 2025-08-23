package yolotriton

import (
	"image"

	triton "github.com/dev6699/yolotriton/grpc-client"
)

type Yolo struct {
	YoloTritonConfig
	metadata struct {
		scaleFactorW float32
		scaleFactorH float32
	}
	io16 bool
}

func NewYolo(cfg YoloTritonConfig, io16 bool) Model {
	return &Yolo{
		YoloTritonConfig: cfg,
		io16:             io16,
	}
}

var _ Model = &Yolo{}

func (y *Yolo) GetConfig() YoloTritonConfig {
	return y.YoloTritonConfig
}

func (y *Yolo) PreProcess(img image.Image, targetWidth uint, targetHeight uint) (*triton.InferTensorContents, error) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()

	preprocessedImg := resizeImage(img, targetWidth, targetHeight)

	y.metadata.scaleFactorW = float32(width) / float32(targetWidth)
	y.metadata.scaleFactorH = float32(height) / float32(targetHeight)

	if y.io16 {
		bytesContents := imageToFloat16ByteSlice(preprocessedImg)
		contents := &triton.InferTensorContents{
			BytesContents: bytesContents,
		}
		return contents, nil
	} else {
		fp32Contents := imageToFloat32Slice(preprocessedImg)
		contents := &triton.InferTensorContents{
			Fp32Contents: fp32Contents,
		}
		return contents, nil
	}
}

func (y *Yolo) PostProcess(rawOutputContents [][]byte) ([]Box, error) {
	var output []float32
	var err error
	if y.io16 {
		output, err = bytesFP16ToFloat32Slice(rawOutputContents[0])
		if err != nil {
			return nil, err
		}
	} else {
		output, err = bytesToFloat32Slice(rawOutputContents[0])
		if err != nil {
			return nil, err
		}
	}

	numObjects := y.NumObjects
	numClasses := y.NumClasses

	boxes := []Box{}

	for index := 0; index < numObjects; index++ {
		classID := 0
		prob := float32(0.0)

		for col := 0; col < numClasses; col++ {
			p := output[numObjects*(col+4)+index]
			if p > prob {
				prob = p
				classID = col
			}
		}

		if prob < y.MinProbability {
			continue
		}

		label := y.Classes[classID]
		x1raw := output[index]
		y1raw := output[numObjects+index]
		w := output[2*numObjects+index]
		h := output[3*numObjects+index]

		x1 := (x1raw - w/2) * y.metadata.scaleFactorW
		y1 := (y1raw - h/2) * y.metadata.scaleFactorH
		x2 := (x1raw + w/2) * y.metadata.scaleFactorW
		y2 := (y1raw + h/2) * y.metadata.scaleFactorH

		boxes = append(boxes, Box{
			X1:          float64(x1),
			Y1:          float64(y1),
			X2:          float64(x2),
			Y2:          float64(y2),
			Probability: float64(prob),
			Class:       label,
		})
	}

	return boxes, nil
}
