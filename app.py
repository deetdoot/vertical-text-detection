import os
import shutil
import uuid
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from process_segments import find_best_ocr_rotation
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PORT = int(os.getenv("PORT", 8000))
TEMP_DIR = os.getenv("TEMP_DIR", "static/temp")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "detected_segments")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "runs/obb/train2/weights/last.pt")
DELETE_FREUENCY = int(os.getenv("DELETE_FREUENCY", 60))

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def schedule_delete_temp(uid: str):
    temp_path = os.path.join(TEMP_DIR, uid)
    time.sleep(DELETE_FREUENCY)
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)

@app.post("/detect_vertical_text")
async def detect_vertical_text(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...)
):
    uid = str(uuid.uuid4())
    req_dir = os.path.join(TEMP_DIR, uid)
    os.makedirs(req_dir, exist_ok=True)
    results = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".png", ".jpg", ".jpeg"]:
            continue
        img_path = os.path.join(req_dir, file.filename)
        contents = await file.read()
        with open(img_path, "wb") as f:
            f.write(contents)
        model = YOLO(YOLO_WEIGHTS)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        detection_results = model(img_np, conf=0.05, verbose=False)
        crops = []
        for result in detection_results:
            xywhr = result.obb.xywhr
            class_ids = result.obb.cls.int()
            names = [result.names[cls.item()] for cls in class_ids]
            for i, name in enumerate(names):
                box = xywhr[i].cpu().numpy() if hasattr(xywhr[i], 'cpu') else xywhr[i]
                box = [float(v) for v in box]
                cx, cy, w, h, angle = box
                angle_deg = angle * 180.0 / np.pi
                rect = ((cx, cy), (w, h), angle_deg)
                box_points = cv2.boxPoints(rect)
                box_points = np.int32(box_points)
                x, y, w_rect, h_rect = cv2.boundingRect(box_points)
                crop_img = img_np[y:y+h_rect, x:x+w_rect].copy()
                mask = np.zeros_like(crop_img)
                pts = box_points - [x, y]
                cv2.drawContours(mask, [pts], 0, (255,255,255), -1)
                crop_img_masked = cv2.bitwise_and(crop_img, mask)
                crop_filename = f"crop_{i}_{name}.png"
                crop_path = os.path.join(req_dir, crop_filename)
                cv2.imwrite(crop_path, crop_img_masked)
                crop_pil = Image.fromarray(cv2.cvtColor(crop_img_masked, cv2.COLOR_BGR2RGB))
                buf = BytesIO()
                crop_pil.save(buf, format='PNG')
                buf.seek(0)
                best_text, best_angle = find_best_ocr_rotation(buf)
                crop_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                crops.append({
                    "crop_image": crop_base64,
                    "ocr_text": best_text,
                    "angle": best_angle,
                    #"class": name
                })
        results.append({
            "filename": file.filename,
            "crops": crops
        })
    background_tasks.add_task(schedule_delete_temp, uid)
    return JSONResponse({"uid": uid, "results": results})

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})