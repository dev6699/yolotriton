# Release 0.4.0
## Breaking Changes
* `Model.GetClass` has been removed in favor of new `YoloTritonConfig.Classes`
* Predefined YOLO model init function now receive `YoloTritonConfig` as argument.
## Major Features and Improvements
* Add `-p <MinProbability>`, `-o <MaxIOU>` flag for sample script.

# Release 0.3.0
## Breaking Changes
* `YoloTritonConfig.NumChannels` has been renamed to `YoloTritonConfig.NumClasses`.
* `YoloTritonConfig.BatchSize` has been removed.
* `Model.PreProcess` interface has been changed to `PreProcess(img image.Image, targetWidth uint, targetHeight uint) (*triton.InferTensorContents, error)`
## Major Features and Improvements
* Add support for YOLO-NAS INT8 inference.
* Add benchmark script.
* Model metadata request will only get once instead everytime before model inference request.

# Release 0.2.0
## Major Features and Improvements
* Add support for YOLO-NAS inference.

# Release 0.1.1
## Bug Fixes and Other Changes
* Fix go pkg publish issue.

# Release 0.1.0
Initial release of yolotriton.
* Support for YOLOv8 inference.