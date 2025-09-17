"""
Vertical Text Detector for Document Images
=========================================

Author: Emtiaz Ahamed Emon
Date: September 17, 2025

This script detects vertical text regions in document images using a YOLO Oriented Bounding Box (OBB) model.

Features:
- Loads a trained YOLO OBB model for vertical text detection.
- Processes all images in the 'gemini-validation-photos' directory.
- For each detected vertical text region, extracts the rotated bounding box coordinates.
- Crops the detected region using the rotated box and exports it to the output folder ('detected_segments_rotated').
- Prints class name, rotated box coordinates, and confidence score for each detection.

Usage:
    python detector.py

Dependencies:
- ultralytics
- opencv-python
- matplotlib
- pillow
- numpy

Edit the paths to weights and images as needed for your project structure.
"""

import os
from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# Path to weights and images
yolo_weights = 'runs/obb/train2/weights/last.pt'
images_dir = 'gemini-validation-photos/'

# Load model
model = YOLO(yolo_weights)

# Get all image files in the directory
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)
    pil_img = Image.open(img_path).convert("L")
    results = model(pil_img, conf=0.05, verbose=True)
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    results = model(img_path, conf=0.05, verbose=True)
    img = cv2.imread(img_path)
    for result in results:
        xywhr = result.obb.xywhr  # (N, 5): center-x, center-y, width, height, angle
        class_ids = result.obb.cls.int()
        names = [result.names[cls.item()] for cls in class_ids]
        confs = result.obb.conf
        result.show()
        for i, name in enumerate(names):
            # Convert tensor to float values
            box = xywhr[i].cpu().numpy() if hasattr(xywhr[i], 'cpu') else xywhr[i]
            box = [float(v) for v in box]
            conf = float(confs[i])
            print(f"Class: {name}")
            print(f"  Rotated box (xywhr): {box}")
            print(f"  Confidence: {conf}")

            # Crop using rotated rectangle
            cx, cy, w, h, angle = box
            # Convert angle from radians to degrees
            angle_deg = angle * 180.0 / np.pi
            rect = ((cx, cy), (w, h), angle_deg)
            # Get box points
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)

            # Get bounding rect and crop
            x, y, w_rect, h_rect = cv2.boundingRect(box_points)
            crop_img = img[y:y+h_rect, x:x+w_rect].copy()

            # Create mask for rotated rectangle
            mask = np.zeros_like(crop_img)
            pts = box_points - [x, y]
            cv2.drawContours(mask, [pts], 0, (255,255,255), -1)
            crop_img_masked = cv2.bitwise_and(crop_img, mask)

            # Save cropped image
            crop_filename = f"detected_segments_rotated/{os.path.splitext(img_name)[0]}_rotcrop_{i}_{name}.jpg"
            cv2.imwrite(crop_filename, crop_img_masked)

    break