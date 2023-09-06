package yolotriton

import (
	"image"
)

type YoloV8 struct {
	YoloTritonConfig
	metadata struct {
		scaleFactorW float32
		scaleFactorH float32
	}
}

func NewYoloV8(modelName string, modelVersion string) Model {
	return &YoloV8{
		YoloTritonConfig: YoloTritonConfig{
			BatchSize:      1,
			NumChannels:    84,
			NumObjects:     8400,
			MinProbability: 0.5,
			MaxIOU:         0.7,
			ModelName:      modelName,
			ModelVersion:   modelVersion,
		},
	}
}

var _ Model = &YoloV8{}

func (y *YoloV8) GetConfig() YoloTritonConfig {
	return y.YoloTritonConfig
}

func (y *YoloV8) PreProcess(img image.Image, targetWidth uint, targetHeight uint) ([]float32, error) {
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()

	preprocessedImg := resizeImage(img, targetWidth, targetHeight)

	fp32Contents := imageToFloat32Slice(preprocessedImg)

	y.metadata.scaleFactorW = float32(width) / float32(targetWidth)
	y.metadata.scaleFactorH = float32(height) / float32(targetHeight)

	return fp32Contents, nil
}

func (y *YoloV8) PostProcess(rawOutputContents [][]byte) ([]Box, error) {
	output, err := bytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}

	numObjects := y.NumObjects
	numChannels := y.NumChannels

	boxes := []Box{}

	for index := 0; index < numObjects; index++ {
		classID := 0
		prob := float32(0.0)

		for col := 0; col < numChannels-4; col++ {
			p := output[numObjects*(col+4)+index]
			if p > prob {
				prob = p
				classID = col
			}
		}

		if prob < y.MinProbability {
			continue
		}

		label := yoloClasses[classID]
		xc := output[index]
		yc := output[numObjects+index]
		w := output[2*numObjects+index]
		h := output[3*numObjects+index]

		x1 := (xc - w/2) * y.metadata.scaleFactorW
		y1 := (yc - h/2) * y.metadata.scaleFactorH
		x2 := (xc + w/2) * y.metadata.scaleFactorW
		y2 := (yc + h/2) * y.metadata.scaleFactorH

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
