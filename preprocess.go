package yolotriton

import (
	"image"
	"image/color"

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
