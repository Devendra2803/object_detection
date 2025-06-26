# Object Detector FastAPI 🎯🎥

This project provides FastAPI-based APIs for running object detection on uploaded videos using:
- 📦 MobileNet SSD (OpenCV DNN)
- 🧠 Faster R-CNN (PyTorch torchvision)
- ⚡ SSD TensorFlow Model (via OpenCV DNN)

All models process the video frame-by-frame, draw bounding boxes with class labels, and return the path to the processed video along with basic evaluation metrics (precision, recall, etc.).

---

## ⚙️ Setup Instructions

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt

2. 🚀 Run the Detector APIs
Make sure your output_videos/ folder exists — it will be auto-created if not.

▶️ MobileNet SSD API
bash

python mobilenet_ssd_api.py
# Runs at: http://127.0.0.1:8001
▶️ SSD TensorFlow API

python ssd_tensorflow_api.py
# Runs at: http://127.0.0.1:8002
▶️ Faster R-CNN API

python frcnn_api.py
# Runs at: http://127.0.0.1:8003
📬 API Endpoint
POST /detect-video-[model]

Upload a .mp4 video via form-data (file)

Returns path to processed video and mock metrics

object_detector/
├── frcnn_api.py             # Faster R-CNN API (port 8003)
├── mobilenet_ssd_api.py     # MobileNet SSD API (port 8001)
├── ssd_tensorflow_api.py    # SSD TensorFlow API (port 8002)
├── requirements.txt         # Dependencies
├── README.md                # This file
└── output_videos/           # Processed videos will be saved here
