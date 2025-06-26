# Object Detector FastAPI 🎯🎥

This project provides FastAPI-based APIs for running object detection on uploaded videos using:
- 📦 MobileNet SSD (OpenCV DNN)
- 🧠 Faster R-CNN (PyTorch torchvision)
- ⚡ SSD TensorFlow Model (via OpenCV DNN)
- 🧠 YOLOv8 (Ultralytics)

All models process the video frame-by-frame, draw bounding boxes with class labels, and return metadata along with storing the processed video **directly in PostgreSQL** (as byte format), and evaluation metrics like **precision**, **recall**, and **frame count**.

---

## ⚙️ Setup Instructions

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
2. 🚀 Run the Detector APIs
Make sure your output_videos/ folder exists — it will be auto-created if not.

Each model runs on a separate port:

bash
Copy
Edit
# SSD
uvicorn ssdmodel:app --reload --port 8001

# FRCNN
uvicorn frcnnmodel:app --reload --port 8002

# YOLOv8
uvicorn yolov:app --reload --port 8003
🧪 API Endpoints
Each model exposes the same 3 routes:

Method	Endpoint	Description
POST	/detect-video-<model>	Upload and process video
GET	/get-output-<model>/{record_id}	Get metadata by ID
GET	/download-output-<model>/{record_id}	Download video output by ID

Replace <model> with: ssd, frcnn, or yolo

Example:

bash
Copy
Edit
http://127.0.0.1:8003/docs     # YOLOv8 Swagger
🗃️ Database Schema
All detection results are stored in a PostgreSQL table:

Table: detection_model_training

Field	Type	Description
id	Integer	Primary key
project_name	String	Example: "Object Detection"
task_name	String	Example: "Video Detection"
model_name	String	Model used
model_path	String	Relative path to model
raw_data_zip	String	Uploaded video name
num_classes	Integer	Number of detected classes
class_names	Text	All class names (comma-separated)
data_size	Integer	Number of frames processed
splitted_data	String	Reserved
best_model_save	String	Note (e.g., "Stored in DB")
video_output	LargeBinary	Byte-encoded processed video
status	String	Default: "Completed"
created_at	DateTime	Auto timestamp

📁 Project Structure
bash
Copy
Edit
object_detector/
│
├── db.py                     # DB connection
├── models.py                 # SQLAlchemy schema
├── frcnnmodel.py             # Faster R-CNN API
├── yolov.py                  # YOLOv8 API
├── ssdmodel.py               # SSD API
├── create_table.py           # Table creation
├── requirements.txt
├── readme.md
├── models/                   # Contains yolov8.pt, .pb, .pbtxt, coco.names
└── output_videos/           # Temporarily used (auto-created if not exists)
