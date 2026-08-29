# 🍓 Fruit Fly Eye Detection

## 📌 Overview

This project uses **computer vision and deep learning** to detect and identify the eyes of fruit flies from images. The system is designed to automatically locate fruit fly eyes using an object detection model, reducing the need for manual identification and enabling faster analysis of biological images.

The project uses **YOLO (You Only Look Once)** for object detection and provides a foundation for automated fruit fly image analysis.

## 🎯 Project Objectives

The main objectives of this project are to:

* Detect fruit flies within images.
* Identify and localise fruit fly eyes.
* Use deep learning to automate eye detection.
* Reduce the time required for manual image analysis.
* Provide accurate bounding-box predictions around detected eyes.
* Create a system that can be extended for further biological image analysis.

## 🧠 Technology

The project is built using:

* **Python**
* **YOLO / Ultralytics**
* **OpenCV**
* **PyTorch**
* **NumPy**
* **Matplotlib**

### Model

The project uses a **YOLO object detection model**, which is trained on an annotated fruit fly dataset.

The model learns to identify the visual characteristics of fruit fly eyes and predicts their locations using bounding boxes.

## 📂 Project Structure

```text
Fruit-Fly-Eye-Detection/
│
├── dataset/
│   ├── images/
│   ├── labels/
│   └── data.yaml
│
├── runs/
│   └── detect/
│
├── models/
│   └── best.pt
│
├── notebooks/
│   └── analysis.ipynb
│
├── detect.py
├── train.py
├── requirements.txt
└── README.md
```

> The exact structure may vary depending on the version of the project.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd Fruit-Fly-Eye-Detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, the main dependencies can be installed with:

```bash
pip install ultralytics opencv-python numpy matplotlib torch
```

## 🏋️ Model Training

The YOLO model can be trained using the annotated dataset.

Example:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640
)
```

The trained model can then be saved and used for inference.

## 🔍 Detection

To run detection on an image:

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")

results = model("example.jpg")

results[0].show()
```

The model will generate bounding boxes around detected fruit fly eyes.

## 📊 Model Evaluation

The trained model can be evaluated using standard object detection metrics, including:

* **Precision**
* **Recall**
* **mAP@50**
* **mAP@50–95**

These metrics can be used to assess how accurately the model identifies and localises fruit fly eyes.

## 🖼️ Example Output

The expected output is an image containing bounding boxes around the detected fruit fly eyes, along with the model's confidence score.

```text
Input Image
     ↓
YOLO Object Detection Model
     ↓
Fruit Fly Eye Detection
     ↓
Bounding Box + Confidence Score
```

## 🚀 Future Improvements

Potential improvements to the project include:

* Increasing the size and diversity of the training dataset.
* Improving detection accuracy for small or partially obscured eyes.
* Experimenting with different YOLO architectures.
* Applying image augmentation during training.
* Detecting additional fruit fly features.
* Developing a real-time detection application.
* Deploying the trained model through a web application.

## 👨‍💻 Author

**Alessio Maggio**

MSc Artificial Intelligence

---

## 📄 License

This project is intended for educational and research purposes.
