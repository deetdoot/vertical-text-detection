# Vertical Text Detection in Document Images

Author: Emtiaz Ahamed Emon  
Date: September 17, 2025

## Overview
This project uses a YOLO Oriented Bounding Box (OBB) model to detect vertical text regions in scanned document images. Detected regions are cropped using the rotated bounding box and exported to an output folder for further processing or review.

## Features
- Detects vertical text in document images using YOLO OBB.
- Crops detected regions using rotated bounding boxes.
- Exports cropped regions to `detected_segments_rotated/`.
- Prints class name, rotated box coordinates, and confidence score for each detection.

## Usage
1. Place your document images in the `gemini-validation-photos/` folder.
2. Run the detector script:
   ```bash
   python detector.py
   ```
3. Cropped vertical text regions will be saved in `detected_segments_rotated/`.

## Example Screenshot
Below is an example of a detected vertical text region (with bounding box overlay):

**Original Image**
<img width="774" height="1000" alt="87594142_87594144" src="https://github.com/user-attachments/assets/2e0bdd40-f7c3-4488-9c1e-6b8198f8fda5" />

**Detection:**
<img width="600" height="803" alt="image" src="https://github.com/user-attachments/assets/007b5cef-aa8d-4918-baa1-22f1e396a3b3" />


**Cropped Section:**
![87594142_87594144_rotcrop_0_vertical-slanted](https://github.com/user-attachments/assets/235a3837-e985-419a-94e6-5515e17d06d5)

*Replace `example_detection.jpg` with a real output from your project after running the script.*

## Training
To train your own model, use the provided `train_model.py` script:
```python
from ultralytics import YOLO
model = YOLO("yolo11n-obb.pt")
results = model.train(data="first_run.yaml", epochs=100, imgsz=640)
```

## Dependencies
- ultralytics
- opencv-python
- matplotlib
- pillow
- numpy

## Credits
Developed by Emtiaz Ahamed Emon
