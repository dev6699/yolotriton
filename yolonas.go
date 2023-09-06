package yolotriton

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

type YoloNAS struct {
	YoloTritonConfig
	metadata struct {
		xOffset     float32
		yOffset     float32
		scaleFactor float32
	}
}

func NewYoloNAS(modelName string, modelVersion string) Model {
	return &YoloNAS{
		YoloTritonConfig: YoloTritonConfig{
			BatchSize:      1,
			NumChannels:    80,
			NumObjects:     8400,
			MinProbability: 0.5,
			MaxIOU:         0.7,
			ModelName:      modelName,
			ModelVersion:   modelVersion,
		},
	}
}

var _ Model = &YoloNAS{}

func (y *YoloNAS) GetConfig() YoloTritonConfig {
	return y.YoloTritonConfig
}

func (y *YoloNAS) PreProcess(img image.Image, targetWidth uint, targetHeight uint) ([]float32, error) {
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

	return fp32Contents, nil
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

		for col := 0; col < y.NumChannels; col++ {
			p := predScores[index*y.NumChannels+(col)]
			if p > prob {
				prob = p
				classID = col
			}
		}

		if prob < y.MinProbability {
			continue
		}

		label := yoloClasses[classID]
		i := (index * 4)
		xc := predBoxes[i]
		yc := predBoxes[i+1]
		w := predBoxes[i+2]
		h := predBoxes[i+3]

		scale := y.metadata.scaleFactor
		x1 := (xc - y.metadata.xOffset) / scale
		y1 := (yc - y.metadata.yOffset) / scale
		x2 := (w - y.metadata.xOffset) / scale
		y2 := (h - y.metadata.yOffset) / scale

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

func padImageToCenterWithGray(originalImage image.Image, targetWidth, targetHeight int, grayValue uint8) (image.Image, int, int) {
	// Calculate the dimensions of the original image
	originalWidth := originalImage.Bounds().Dx()
	originalHeight := originalImage.Bounds().Dy()

	// Calculate the padding dimensions
	padWidth := targetWidth - originalWidth
	padHeight := targetHeight - originalHeight

	// Create a new RGBA image with the desired dimensions and fill it with gray
	paddedImage := image.NewRGBA(image.Rect(0, 0, targetWidth, targetHeight))
	grayColor := color.RGBA{grayValue, grayValue, grayValue, 255}
	draw.Draw(paddedImage, paddedImage.Bounds(), &image.Uniform{grayColor}, image.Point{}, draw.Src)

	// Calculate the position to paste the original image in the center
	xOffset := int(math.Floor(float64(padWidth) / 2))
	yOffset := int(math.Floor(float64(padHeight) / 2))

	// Paste the original image onto the padded image
	pasteRect := image.Rect(xOffset, yOffset, xOffset+originalWidth, yOffset+originalHeight)
	draw.Draw(paddedImage, pasteRect, originalImage, image.Point{}, draw.Over)

	return paddedImage, xOffset, yOffset
}
