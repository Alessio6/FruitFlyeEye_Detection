from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/Users/alessiomaggio/Msc Artificial Intelligence/Semester 2/Group software project/Test/fly.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
#results = model("path/to/bus.jpg")