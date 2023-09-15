package yolotriton

import (
	"image"
	"math"

	triton "github.com/dev6699/yolotriton/grpc-client"
)

type YoloNASInt8 struct {
	YoloTritonConfig
	metadata struct {
		xOffset     float32
		yOffset     float32
		scaleFactor float32
	}
}

func NewYoloNASInt8(modelName string, modelVersion string) Model {
	return &YoloNASInt8{
		YoloTritonConfig: YoloTritonConfig{
			MinProbability: 0.5,
			MaxIOU:         0.7,
			ModelName:      modelName,
			ModelVersion:   modelVersion,
		},
	}
}

var _ Model = &YoloNAS{}

func (y *YoloNASInt8) GetConfig() YoloTritonConfig {
	return y.YoloTritonConfig
}

func (y *YoloNASInt8) GetClass(index int) string {
	return yoloClasses[index]
}

func (y *YoloNASInt8) PreProcess(img image.Image, targetWidth uint, targetHeight uint) (*triton.InferTensorContents, error) {
	height := img.Bounds().Dy()
	width := img.Bounds().Dx()

	scaleFactor := math.Min(float64(636)/float64(height), float64(636)/float64(width))
	if scaleFactor != 1.0 {
		newHeight := uint(math.Round(float64(height) * scaleFactor))
		newWidth := uint(math.Round(float64(width) * scaleFactor))
		img = resizeImage(img, newWidth, newHeight)
	}

	paddedImage, xOffset, yOffset := padImageToCenterWithGray(img, int(targetWidth), int(targetHeight), 114)
	uint32Contents := imageToUint32Slice(paddedImage)

	y.metadata.xOffset = float32(xOffset)
	y.metadata.yOffset = float32(yOffset)
	y.metadata.scaleFactor = float32(scaleFactor)

	contents := &triton.InferTensorContents{
		UintContents: uint32Contents,
	}
	return contents, nil
}

func (y *YoloNASInt8) PostProcess(rawOutputContents [][]byte) ([]Box, error) {
	numPreds, err := bytesToInt32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}
	predBoxes, err := bytesToFloat32Slice(rawOutputContents[1])
	if err != nil {
		return nil, err
	}
	predScores, err := bytesToFloat32Slice(rawOutputContents[2])
	if err != nil {
		return nil, err
	}
	predClasses, err := bytesToInt32Slice(rawOutputContents[3])
	if err != nil {
		return nil, err
	}

	boxes := []Box{}
	detectedObjects := int(numPreds[0])
	for index := 0; index < detectedObjects; index++ {

		prob := predScores[index]
		if prob < y.MinProbability {
			continue
		}

		classID := predClasses[index]
		label := y.GetClass(int(classID))
		idx := (index * 4)
		x1raw := predBoxes[idx]
		y1raw := predBoxes[idx+1]
		x2raw := predBoxes[idx+2]
		y2raw := predBoxes[idx+3]

		scale := y.metadata.scaleFactor
		x1 := (x1raw - y.metadata.xOffset) / scale
		y1 := (y1raw - y.metadata.yOffset) / scale
		x2 := (x2raw - y.metadata.xOffset) / scale
		y2 := (y2raw - y.metadata.yOffset) / scale

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
