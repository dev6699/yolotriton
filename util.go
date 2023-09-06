package yolotriton

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"os"

	"github.com/golang/freetype/truetype"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/math/fixed"
)

func LoadImage(imagePath string) (image.Image, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	return img, nil
}

func SaveImage(img image.Image, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	err = jpeg.Encode(file, img, nil)
	if err != nil {
		return err
	}
	return nil
}

func DrawBoundingBoxes(img image.Image, boxes []Box, lineWidth int, fontSize float64) (image.Image, error) {

	// Create a new RGBA image to draw the bounding boxes and text labels on
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)

	// Copy the original image to the destination image
	draw.Draw(dst, bounds, img, bounds.Min, draw.Over)

	// Create a color for the bounding boxes (red in this example)
	red := color.RGBA{255, 0, 0, 255}

	// Create a font from a TrueType font file with the specified font size
	ttfFont, err := truetype.Parse(goregular.TTF)
	if err != nil {
		return nil, err
	}
	face := truetype.NewFace(ttfFont, &truetype.Options{
		Size: fontSize,
	})

	// Draw the bounding boxes and text labels on the destination image
	for _, box := range boxes {
		x1, y1, x2, y2 := box.X1, box.Y1, box.X2, box.Y2
		// Draw the bounding box
		for x := x1; x <= x2; x++ {
			for w := 0; w < lineWidth; w++ {
				dst.Set(int(x), int(y1)+w, red)
				dst.Set(int(x), int(y2)+w, red)
			}
		}
		for y := y1; y <= y2; y++ {
			for w := 0; w < lineWidth; w++ {
				dst.Set(int(x1)+w, int(y), red)
				dst.Set(int(x2)+w, int(y), red)
			}
		}

		// Draw the text label above the box
		label := fmt.Sprintf("%s %f", box.Class, box.Probability)
		textX := int(x1)
		textY := int(y1) - 5

		d := &font.Drawer{
			Dst:  dst,
			Src:  image.NewUniform(red),
			Face: face,
			Dot:  fixed.P(textX, textY),
		}
		d.DrawString(label)
	}

	return dst, nil
}
