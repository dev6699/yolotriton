package yolotriton

import (
	"image"
	"image/color"
	"image/draw"
	"math"

	"github.com/nfnt/resize"
)

func resizeImage(img image.Image, width, heigth uint) image.Image {
	return resize.Resize(width, heigth, img, resize.Lanczos3)
}

func pixelRGBA(c color.Color) (r, g, b, a uint32) {
	r, g, b, a = c.RGBA()
	return r >> 8, g >> 8, b >> 8, a >> 8
}

func imageToFloat32Slice(img image.Image) []float32 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	inputContents := make([]float32, width*height*3)

	idx := 0
	offset := (height * width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixel := img.At(x, y)
			r, g, b, _ := pixelRGBA(pixel)

			// Normalize the color values to the range [0, 1]
			floatR := float32(r) / 255
			floatG := float32(g) / 255
			floatB := float32(b) / 255

			inputContents[idx] = floatR
			inputContents[offset+idx] = floatG
			inputContents[2*offset+idx] = floatB
			idx++
		}
	}

	return inputContents
}

func imageToUint32Slice(img image.Image) []uint32 {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	inputContents := make([]uint32, width*height*3)

	idx := 0
	offset := (height * width)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			pixel := img.At(x, y)
			r, g, b, _ := pixelRGBA(pixel)

			inputContents[idx] = r
			inputContents[offset+idx] = g
			inputContents[2*offset+idx] = b
			idx++
		}
	}

	return inputContents
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
