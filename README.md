# SqueezeDetTRT

This repository contains an implementation of SqueezeDet, a ["unified, small, low power fully convolutional neural network for real-time object detection for autonomous driving"](https://arxiv.org/abs/1612.01051), in TensorRT and CUDA for inference acceleration.

You can find the original squeezeDet implementation, using Tensorflow, [here](https://github.com/BichenWuUCB/squeezeDet).

## Prerequisites
This requires Ubuntu 16 (Xenial Xeres) or later to get the proper libraries work. It should work for other distributions, but I haven't tested them yet. From a terminal, execute the following:
```
sudo apt-get install build-essential python perl
```
You also have to install OpenCV 3.3 or higher according to its website [here](https://docs.opencv.org/3.3.1/d7/d9f/tutorial_linux_install.html).

And you also have to install CUDA 8.0 and TensorRT 3.0 libraries, according to their websites [CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-80-ga2-download-archive)
and [TensorRT](https://developer.nvidia.com/rdp/form/tensorrt-download-survey). Also remember to put `nvcc` (usually in `/usr/local/cuda/bin`) in environment variable `PATH`.

## Build
Use `make` from a terminal in this folder to compile executable binary.

## Usage
After compilation, use `./sqdtrt -h` to learn the usage for this program, as follows.
```
Usage: sqdtrt [options] IMAGE_DIR RESULT_DIR
Apply SqueezeDet detection algorithm to images in IMAGE_DIR.
Print detection results to one text file per image in RESULT_DIR using KITTI dataset format.

Options:
       -e, --eval-list=EVAL_LIST_FILE          Provide an evaluation list file which contains
                                               the image names (without extension names)
                                               in IMAGE_DIR to be evaluated.
       -v, --video=VIDEO_FILE                  Detect a video file and play detected video
                                               in a new window. IMAGE_DIR and RESULT_DIR
                                               are not needed.
       -b, --bbox-dir=BBOX_DIR                 Draw bounding boxes in images or video and
                                               save them in BBOX_DIR.
       -x, --x-shift=X_SHIFT                   Shift all bboxes downward X_SHIFT pixels.
       -y, --y-shift=Y_SHIFT                   Shift all bboxes rightward Y_SHIFT pixels.
       -h, --help                              Print this help and exit.
```

## Demo
There are two demoes for image and video detections below. The image and video for demoes are located in `data/example`.

### Image detection
The following command will detect an image in `data/example` and print bounding boxes using KITTI format in `data/result/sample.txt`
```
./sqdtrt -e data/example/val.txt data/example data/result
```

### Video detection
The following command will detect a video in `data/example` and play it with bounding boxes in a new window.
There is some slight dislocation due to the image resize operation (maybe?), so we have to use the `-x '-20' -y '-20'` arguments to fix it.
```
./sqdtrt -v data/example/20110926.avi -x '-20' -y '-20'
```

