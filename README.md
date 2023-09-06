# yolotriton

[![GoDoc](https://pkg.go.dev/badge/github.com/dev6699/yolotriton)](https://pkg.go.dev/github.com/dev6699/yolotriton)
[![Go Report Card](https://goreportcard.com/badge/github.com/dev6699/yolotriton)](https://goreportcard.com/report/github.com/dev6699/yolotriton)
[![License](https://img.shields.io/github/license/dev6699/yolotriton)](LICENSE)

Go (Golang) gRPC client for YOLO-NAS, YOLOv8 inference using the Triton Inference Server.

## Installation

Use `go get` to install this package:

```bash
go get github.com/dev6699/yolotriton
```

### Get YOLO-NAS, YOLOv8 TensorRT model
Replace `yolov8m.pt` with your desired model
```bash
pip install ultralytics
yolo export model=yolov8m.pt format=onnx
trtexec --onnx=yolov8m.onnx --saveEngine=model_repository/yolov8/1/model.plan
```

References:
1. https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html
2. https://docs.ultralytics.com/modes/export/
3. https://github.com/NVIDIA/TensorRT/tree/master/samples/trtexec

### Start trinton server
```bash
docker compose up tritonserver
```
References:
1. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html

### Sample usage
Check [cmd/main.go](cmd/main.go) for more details.

Available args:
```bash
  -i string
        Inference Image. (default "images/1.jpg")
  -m string
        Name of model being served (Required) (default "yolonas")
  -t string
        Type of model. Available options: [yolonas, yolov8] (default "yolonas")
  -u string
        Inference Server URL. (default "tritonserver:8001")
  -x string
        Version of model. Default: Latest Version
```
```bash
go run cmd/main.go
```

### Results
```
prediction:  0
class:  dog
confidence: 0.96
bboxes: [ 669 130 1061 563 ]
---------------------
prediction:  1
class:  person
confidence: 0.96
bboxes: [ 440 30 760 541 ]
---------------------
prediction:  2
class:  dog
confidence: 0.93
bboxes: [ 168 83 495 592 ]
---------------------
```
| Input                       | YOLO-NAS Ouput                          | YOLOv8 Output                          |
| --------------------------- | --------------------------------------- | -------------------------------------- |
| <img src="images/1.jpg" />  | <img src="images/1_yolonas_out.jpg" />  | <img src="images/1_yolonas_out.jpg" /> |
| <img src="images/2.jpg" />  | <img src="images/2_yolonas_out.jpg" />  | <img src="images/2_yolonas_out.jpg" /> |