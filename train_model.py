from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n-obb.pt")

# Train the model
results = model.train(data="first_run.yaml", epochs=100, imgsz=640)