AI-Based Road Inspection System

🚧 Project Overview

The AI-Based Road Inspection System leverages deep learning and computer vision techniques to detect and classify road defects such as potholes, cracks, and other anomalies. It is built using the YOLOv8 architecture for object detection and includes a Streamlit-based web interface for user interaction.

🗂️ Dataset Details

Source: Dataset hosted and annotated on Roboflow.

Contents:

Images of roads annotated with labels for defects such as:

🕳️ Potholes

⚡ Cracks

🐊 Alligator cracks

Annotation Format: YOLO format (bounding boxes with class labels).

Training/Validation Split: Dataset is divided into training, validation, and test sets for effective model evaluation.

🤖 Model Details

Model Used

YOLOv8 (You Only Look Once Version 8)

Known for its speed and accuracy in real-time object detection.

Base model: yolov8s.pt (pre-trained on the COCO dataset).

Training Details

Framework: Ultralytics YOLOv8

Hyperparameters:

🕒 Epochs: 100

📦 Batch Size: 4

🖼️ Image Size: 640 x 640

🎯 Confidence Threshold: 0.25 (for prediction)

Optimization: Fine-tuned on the annotated road defect dataset.

Evaluation Metrics:

Confusion Matrix

Normalized Confusion Matrix

Precision-Recall Curves

Validation Loss

Training Command

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=100 imgsz=640 batch=4

🌐 Frontend Framework

Framework: Streamlit

Provides an intuitive web interface for interacting with the trained model. Users can upload images or videos and view defect predictions in real time.

Key Components

📋 Sidebar Menu:

Project Information

Predict Defects

Model Information

🖼️ Image Processing: Upload and process images for defect detection.

🎥 Video Processing: Upload and process videos to detect defects frame-by-frame.

📊 Output: Visualizations with bounding boxes and labels on processed images/videos.

🛠️ Backend and Core Libraries

YOLOv8 Framework: ultralytics (for training and inference).

Image Processing:

OpenCV

Pillow

Data Visualization:

Matplotlib

Altair

Additional Libraries:

🎬 moviepy: Video handling

🖼️ imageio: Image sequence handling

🖱️ streamlit_option_menu: Sidebar creation

📊 numpy: Numerical computations

💻 Environment and Dependencies

System Requirements

Hardware:

NVIDIA GPU (e.g., P100 or RTX 3050 recommended for model training).

CUDA support for GPU acceleration.

Required Packages

System Packages (packages.txt):

libgl1
libglib2.0-0
ffmpeg

Python Libraries (requirements.txt):

imageio==2.26.1
moviepy==1.0.3
numpy>1.23.5
opencv-python==4.7.0.72
Pillow==10.4.0
streamlit==1.2.0
streamlit_option_menu==0.3.2
ultralytics==8.0.112
protobuf==3.20.1
altair==4
opencv-python-headless

🏗️ Training Process

Data Preparation

Annotated dataset is downloaded using the Roboflow API:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("defect-road-detection").project("detection2-wyo5q")
dataset = project.version(8).download("yolov8")

Model Training

Train the YOLOv8 model on the dataset:

!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=100 imgsz=640 batch=4

Model Evaluation

Evaluate the trained model using validation data:

!yolo task=detect mode=val model=weights/best.pt data={dataset.location}/data.yaml

Inference

Run inference on test images or videos:

!yolo task=detect mode=predict model=weights/best.pt conf=0.25 source={dataset.location}/test/images

📂 Output Files

Model Weights: Saved in the weights/ directory after training (best.pt).

Evaluation Results: Include confusion matrices, precision-recall curves, and result visualizations.

Processed Images/Videos: Saved in the runs/detect/predict/ folder with bounding boxes and labels.

🗂️ Repository Structure

.
├── app.py                 # Streamlit application code
├── training.ipynb         # Notebook for model training
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies
├── README.md              # Project documentation (this file)
├── weights/               # Pre-trained model weights (e.g., best.pt)
├── runs/                  # Folder containing processed results
└── dataset/               # Instructions for downloading the dataset

📝 Notes

Pre-trained weights (best.pt) are included for easy deployment/testing.

Ensure to replace YOUR_API_KEY with your Roboflow API key.

For any issues or questions, feel free to open an issue in the repository.

⚖️ License

This project is licensed under the MIT License. See the LICENSE file for more details.
