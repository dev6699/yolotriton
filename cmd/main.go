package main

import (
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/dev6699/yolotriton"
)

type Flags struct {
	ModelName      string
	ModelVersion   string
	ModelType      string
	URL            string
	Image          string
	Benchmark      bool
	BenchmarkCount int
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "m", "yolonas", "Name of model being served (Required)")
	flag.StringVar(&flags.ModelVersion, "x", "", "Version of model. Default: Latest Version")
	flag.StringVar(&flags.ModelType, "t", "yolonas", "Type of model. Available options: [yolonas, yolonasint8, yolov8]")
	flag.StringVar(&flags.URL, "u", "tritonserver:8001", "Inference Server URL.")
	flag.StringVar(&flags.Image, "i", "images/1.jpg", "Inference Image.")
	flag.BoolVar(&flags.Benchmark, "b", false, "Run benchmark.")
	flag.IntVar(&flags.BenchmarkCount, "n", 1, "Number of benchmark run.")
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
	case yolotriton.ModelTypeYoloNASInt8:
		model = yolotriton.NewYoloNASInt8(FLAGS.ModelName, FLAGS.ModelVersion)
	default:
		log.Fatalf("Unsupported model: %s. Available options: [yolonas, yolonasint8, yolov8]", FLAGS.ModelType)
	}

	yt, err := yolotriton.New(FLAGS.URL, model)
	if err != nil {
		log.Fatal(err)
	}

	img, err := yolotriton.LoadImage(FLAGS.Image)
	if err != nil {
		log.Fatalf("Failed to preprocess image: %v", err)
	}

	loop := 1
	if FLAGS.Benchmark {
		loop = FLAGS.BenchmarkCount
	}

	start := time.Now()
	for i := 0; i < loop; i++ {
		now := time.Now()
		results, err := yt.Infer(img)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%d. processing time: %s\n", i+1, time.Since(now))
		if FLAGS.Benchmark {
			continue
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

	if FLAGS.Benchmark {
		fmt.Println("Avg processing time:", time.Since(start)/time.Duration(FLAGS.BenchmarkCount))
	}
}
