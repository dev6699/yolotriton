package yolotriton

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
)

func bytesToFloat32Slice(data []byte) ([]float32, error) {
	t := []float32{}

	// Create a buffer from the input data
	buffer := bytes.NewReader(data)
	for {
		// Read the binary data from the buffer
		var binaryValue uint32
		err := binary.Read(buffer, binary.LittleEndian, &binaryValue)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		t = append(t, math.Float32frombits(binaryValue))

	}

	return t, nil
}

func bytesToInt32Slice(data []byte) ([]int32, error) {
	t := []int32{}

	// Create a buffer from the input data
	buffer := bytes.NewReader(data)
	for {
		// Read the binary data from the buffer
		var binaryValue uint32
		err := binary.Read(buffer, binary.LittleEndian, &binaryValue)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}

		t = append(t, int32(binaryValue))

	}

	return t, nil
}

type Box struct {
	X1          float64
	Y1          float64
	X2          float64
	Y2          float64
	Probability float64
	Class       string
}

func iou(box1, box2 Box) float64 {
	// Calculate the coordinates of the intersection rectangle
	intersectionX1 := math.Max(box1.X1, box2.X1)
	intersectionY1 := math.Max(box1.Y1, box2.Y1)
	intersectionX2 := math.Min(box1.X2, box2.X2)
	intersectionY2 := math.Min(box1.Y2, box2.Y2)

	// Calculate the area of the intersection rectangle
	intersectionArea := math.Max(0, intersectionX2-intersectionX1+1) * math.Max(0, intersectionY2-intersectionY1+1)

	// Calculate the area of each bounding box
	box1Area := (box1.X2 - box1.X1 + 1) * (box1.Y2 - box1.Y1 + 1)
	box2Area := (box2.X2 - box2.X1 + 1) * (box2.Y2 - box2.Y1 + 1)

	// Calculate the IoU
	iou := intersectionArea / (box1Area + box2Area - intersectionArea)

	return iou
}
