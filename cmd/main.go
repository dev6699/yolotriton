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
	ModelType    string
	URL          string
	Image        string
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "m", "yolonas", "Name of model being served (Required)")
	flag.StringVar(&flags.ModelVersion, "x", "", "Version of model. Default: Latest Version")
	flag.StringVar(&flags.ModelType, "t", "yolonas", "Type of model. Available options: [yolonas, yolov8]")
	flag.StringVar(&flags.URL, "u", "tritonserver:8001", "Inference Server URL.")
	flag.StringVar(&flags.Image, "i", "images/1.jpg", "Inference Image.")
	flag.Parse()
	return flags
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	var model yolotriton.Model
	switch yolotriton.ModelType(FLAGS.ModelType) {
	case yolotriton.ModelTypeYoloV8:
		model = yolotriton.NewYoloV8(FLAGS.ModelName, FLAGS.ModelVersion)
	case yolotriton.ModelTypeYoloNAS:
		model = yolotriton.NewYoloNAS(FLAGS.ModelName, FLAGS.ModelVersion)
	default:
		log.Fatalf("Unsupported model: %s. Available options: [yolonas, yolov8]", FLAGS.ModelType)
	}

	ygt, err := yolotriton.New(FLAGS.URL, model)
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
		fmt.Println("prediction: ", i)
		fmt.Println("class: ", r.Class)
		fmt.Printf("confidence: %.2f\n", r.Probability)
		fmt.Println("bboxes: [", int(r.X1), int(r.Y1), int(r.X2), int(r.Y2), "]")
		fmt.Println("---------------------")
	}

	out, err := yolotriton.DrawBoundingBoxes(
		img,
		results,
		int(float64(img.Bounds().Dx())*0.005),
		float64(img.Bounds().Dx())*0.02,
	)
	if err != nil {
		log.Fatal(err)
	}

	err = yolotriton.SaveImage(out, fmt.Sprintf("%s_%s_out.jpg", strings.Split(FLAGS.Image, ".")[0], FLAGS.ModelName))
	if err != nil {
		log.Fatal(err)
	}
}
