package yolotriton

import (
	"bytes"
	"encoding/binary"
	"math"
	"sort"
)

func (y *YoloTriton) bytesToFloat32Slice(data []byte) ([]float32, error) {
	t := []float32{}

	// Create a buffer from the input data
	buffer := bytes.NewReader(data)
	for i := 0; i < y.cfg.BatchSize; i++ {
		for j := 0; j < y.cfg.NumChannels; j++ {
			for k := 0; k < y.cfg.NumObjects; k++ {
				// Read the binary data from the buffer
				var binaryValue uint32
				err := binary.Read(buffer, binary.LittleEndian, &binaryValue)
				if err != nil {
					return nil, err
				}

				t = append(t, math.Float32frombits(binaryValue))

			}
		}
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

func (y *YoloTriton) parseOutput(output []float32, origImgWidth, origImgHeight int) []Box {
	boxes := []Box{}

	for index := 0; index < y.cfg.NumObjects; index++ {
		classID := 0
		prob := float32(0.0)

		for col := 0; col < y.cfg.NumChannels-4; col++ {
			if output[y.cfg.NumObjects*(col+4)+index] > prob {
				prob = output[y.cfg.NumObjects*(col+4)+index]
				classID = col
			}
		}

		if prob < float32(y.cfg.MinProbability) {
			continue
		}

		label := yoloClasses[classID]
		xc := output[index]
		yc := output[y.cfg.NumObjects+index]
		w := output[2*y.cfg.NumObjects+index]
		h := output[3*y.cfg.NumObjects+index]
		x1 := (xc - w/2) / float32(y.cfg.Width) * float32(origImgWidth)
		y1 := (yc - h/2) / float32(y.cfg.Height) * float32(origImgHeight)
		x2 := (xc + w/2) / float32(y.cfg.Width) * float32(origImgWidth)
		y2 := (yc + h/2) / float32(y.cfg.Height) * float32(origImgHeight)
		boxes = append(boxes, Box{
			X1:          float64(x1),
			Y1:          float64(y1),
			X2:          float64(x2),
			Y2:          float64(y2),
			Probability: float64(prob),
			Class:       label,
		})
	}

	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Probability < boxes[j].Probability
	})

	result := []Box{}
	for len(boxes) > 0 {
		result = append(result, boxes[0])
		tmp := []Box{}
		for _, box := range boxes {
			if iou(boxes[0], box) < y.cfg.MaxIOU {
				tmp = append(tmp, box)
			}
		}
		boxes = tmp
	}

	return result
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
