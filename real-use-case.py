import os
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt

# Path to weights and images
yolo_weights = 'runs/obb/train2/weights/last.pt'
images_dir = 'gemini-validation-photos/'

# Load model
model = YOLO(yolo_weights)

# Get all image files in the directory
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(image_files)

for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)
    results = model(img_path, conf=0.05)
    if not results:
        continue

    img = cv2.imread(img_path)
    for result in results:
        result = result.show()
        # Check if result has boxes
        if hasattr(result, 'boxes') and result.boxes is not None and hasattr(result.boxes, 'xyxy'):
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy
            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf
            class_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls
            names = result.names if hasattr(result, 'names') else None

            for box, conf, cls in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f"{names[int(cls)] if names else int(cls)}: {conf:.2f}"
                cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predictions for {img_name}")
    plt.axis('off')
    plt.show()

