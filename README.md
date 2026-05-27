# Human Detection System

A real-time human detection application built with Python and OpenCV, using YOLOv4-tiny for accurate person detection in images, videos, and live webcam feeds.

---

## Features

- Detects humans in images, pre-recorded videos, and live webcam streams
- Displays bounding boxes with confidence scores around each detected person
- Real-time person count displayed on screen
- Optional output saving for both video and image results
- Soft-NMS support for better detection in crowded/overlapping scenes

---

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- imutils
- NumPy

Install dependencies:
```bash
pip install opencv-python imutils numpy
```

---

## Model Files (Required)

Download these three files and place them in the **same folder** as the script:

| File | Download |
|------|----------|
| `yolov4-tiny.weights` | [Download](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights) |
| `yolov4-tiny.cfg` | [Download](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg) |
| `coco.names` | [Download](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names) |

Or use curl:
```bash
curl -L "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights" -o yolov4-tiny.weights
curl -L "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg" -o yolov4-tiny.cfg
curl -L "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names" -o coco.names
```


---

## Usage

### Webcam (live detection)
```bash
python python_pp.py -c
```

### Video file
```bash
python python_pp.py -v "path\to\video.mp4"
```

### Image file
```bash
python python_pp.py -i "path\to\image.jpg"
```

> **Windows tip:** Always use double quotes `"..."` for paths — single quotes are not supported in Windows Command Prompt.

### Save output to file
Add `-o` with an output path to any of the above commands:
```bash
python python_pp.py -v "path\to\video.mp4" -o "path\to\output.avi"
python python_pp.py -i "path\to\image.jpg" -o "path\to\output.jpg"
```

### All arguments

| Argument | Description |
|----------|-------------|
| `-c` | Use webcam for live detection |
| `-v VIDEO` | Path to a video file |
| `-i IMAGE` | Path to an image file |
| `-o OUTPUT` | Path to save the output file (optional) |

Press **Q** to quit during video or webcam detection.

---

## Project Structure

```
project/
│
├── python_pp.py        # Main script
├── yolov4-tiny.weights # YOLO model weights (download separately)
├── yolov4-tiny.cfg     # YOLO model config (download separately)
└── coco.names          # Class labels (download separately)
```

---

## How It Works

1. Each frame is passed through YOLOv4-tiny, a fast convolutional neural network trained on the COCO dataset.
2. Detections with confidence above 0.3 and class label `person` are kept.
3. Non-Maximum Suppression (NMS) removes duplicate bounding boxes.
4. Bounding boxes and confidence scores are drawn on the frame and displayed.

---

## Tuning Detection

Two parameters in the `detect()` function can be adjusted:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `conf_threshold` | `0.3` | Lower = detects more people but may add false positives |
| `nms_threshold` | `0.3` | Lower = stricter duplicate removal; raise if overlapping people are missed |

For crowded scenes with overlapping people, the script uses **Soft-NMS**, which decays confidence scores of overlapping boxes rather than deleting them outright — this preserves detections of people standing close together.

---

## Known Limitations

- YOLOv4-tiny may miss partially occluded people in dense crowds; use full YOLOv4 for higher accuracy.
- Detection speed depends on CPU — expect 5–15 FPS on most machines without a GPU.
- Very small or distant people may not be detected reliably.

---

## License

This project uses the YOLOv4 model by [AlexeyAB](https://github.com/AlexeyAB/darknet), which is licensed under the MIT License.
