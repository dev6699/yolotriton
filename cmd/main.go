package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/dev6699/yolotriton"
)

type Flags struct {
	ModelName    string
	ModelVersion string
	URL          string
	Image        string
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "m", "yolov8_tensorrt", "Name of model being served. (Required)")
	flag.StringVar(&flags.ModelVersion, "x", "", "Version of model. Default: Latest Version.")
	flag.StringVar(&flags.URL, "u", "tritonserver:8001", "Inference Server URL. Default: tritonserver:8001")
	flag.StringVar(&flags.Image, "i", "images/1.jpg", "Inference Image. Default: images/1.jpg")
	flag.Parse()
	return flags
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	ygt, err := yolotriton.New(
		FLAGS.URL,
		yolotriton.YoloTritonConfig{
			BatchSize:      1,
			NumChannels:    84,
			NumObjects:     8400,
			Width:          640,
			Height:         640,
			ModelName:      FLAGS.ModelName,
			ModelVersion:   FLAGS.ModelVersion,
			MinProbability: 0.5,
			MaxIOU:         0.7,
		})

	if err != nil {
		log.Fatal(err)
	}

	img, err := yolotriton.LoadImage(FLAGS.Image)
	if err != nil {
		log.Fatalf("Failed to preprocess image: %v", err)
	}

	results, err := ygt.Infer(img)
	if err != nil {
		log.Fatal(err)
	}

	for i, r := range results {
		fmt.Printf("---%d---", i)
		fmt.Println(r.Class, r.Probability)
		fmt.Println("[x1,x2,y1,y2]", int(r.X1), int(r.X2), int(r.Y1), int(r.Y2))
	}

	out, err := yolotriton.DrawBoundingBoxes(img, results, 5)
	if err != nil {
		log.Fatal(err)
	}

	err = yolotriton.SaveImage(out, fmt.Sprintf("%s_out.jpg", strings.Split(FLAGS.Image, ".")[0]))
	if err != nil {
		log.Fatal(err)
	}
}
