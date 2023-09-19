package yolotriton

import (
	"image"
	"math"

	triton "github.com/dev6699/yolotriton/grpc-client"
)

type YoloNAS struct {
	YoloTritonConfig
	metadata struct {
		xOffset     float32
		yOffset     float32
		scaleFactor float32
	}
}

func NewYoloNAS(cfg YoloTritonConfig) Model {
	return &YoloNAS{
		YoloTritonConfig: cfg,
	}
}

var _ Model = &YoloNAS{}

func (y *YoloNAS) GetConfig() YoloTritonConfig {
	return y.YoloTritonConfig
}

func (y *YoloNAS) PreProcess(img image.Image, targetWidth uint, targetHeight uint) (*triton.InferTensorContents, error) {
	height := img.Bounds().Dy()
	width := img.Bounds().Dx()

	// https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/processing/processing.py#L547
	scaleFactor := math.Min(float64(636)/float64(height), float64(636)/float64(width))
	if scaleFactor != 1.0 {
		newHeight := uint(math.Round(float64(height) * scaleFactor))
		newWidth := uint(math.Round(float64(width) * scaleFactor))
		img = resizeImage(img, newWidth, newHeight)
	}

	paddedImage, xOffset, yOffset := padImageToCenterWithGray(img, int(targetWidth), int(targetHeight), 114)

	fp32Contents := imageToFloat32Slice(paddedImage)

	y.metadata.xOffset = float32(xOffset)
	y.metadata.yOffset = float32(yOffset)
	y.metadata.scaleFactor = float32(scaleFactor)

	contents := &triton.InferTensorContents{
		Fp32Contents: fp32Contents,
	}
	return contents, nil
}

func (y *YoloNAS) PostProcess(rawOutputContents [][]byte) ([]Box, error) {
	predScores, err := bytesToFloat32Slice(rawOutputContents[0])
	if err != nil {
		return nil, err
	}
	predBoxes, err := bytesToFloat32Slice(rawOutputContents[1])
	if err != nil {
		return nil, err
	}

	boxes := []Box{}

	for index := 0; index < y.NumObjects; index++ {

		classID := 0
		prob := float32(0.0)

		for col := 0; col < y.NumClasses; col++ {
			p := predScores[index*y.NumClasses+(col)]
			if p > prob {
				prob = p
				classID = col
			}
		}

		if prob < y.MinProbability {
			continue
		}

		label := y.Classes[classID]
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
