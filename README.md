# ONNX Runtime + OpenCV Inference (LaMa) — C++ Mini Demo

Small C++ example that loads a model, reads an image and a mask, runs inference, and saves the result as a timestamped PNG. Works with single-image input and supports either 1-channel or 3-channel outputs.

## Prerequisites
- C++17 toolchain (GCC/Clang/MSVC)
- OpenCV (core, imgcodecs, dnn)
- ONNX Runtime (C++ API)
- CMake ≥ 3.16

## Features

- Minimal single-file C++ demo
- Image + mask preprocessing with OpenCV `blobFromImage`
- Zero-copy tensor wrapping for inputs
- ONNX Runtime inference with graph optimizations
- Output handling for `C=1` (grayscale) and `C=3` (color)
- Safe float→8-bit scaling and PNG export (timestamped)
- Basic input/output shape logging and validation
- Error handling with detailed ORT exception messages

## Build
Ensure CMake can locate both OpenCV and ONNX Runtime. If ONNX Runtime isn’t globally discoverable, pass `-Donnxruntime_DIR=/path/to/onnxruntime/cmake` (directory containing `onnxruntime-config.cmake`). For OpenCV, pass `-DOpenCV_DIR=/path/to/opencv`.

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -Donnxruntime_DIR=/path/to/onnxruntime/cmake \
      -DOpenCV_DIR=/path/to/opencv ..
cmake --build . --config Release
```


## Run

Place files here:

```
./models/lama_fp32.onnx

./images/input_image.jpg

./images/dilated_mask.png
```
Then run:
```
./onnx_inference
```

Console will print input/output shapes and save PNG(s) under "./outputs/" with names like:
```
output_HHMMSS_0.png
```