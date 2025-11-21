from fastapi import APIRouter, UploadFile, File, HTTPException, Query
import cv2
import numpy as np
from core.ai_loader import ai_engine
from core.fastapi_util import api_response_data
from core.constants import Result
from PIL import Image
import base64
import io
import torch
from torchvision import transforms

router = APIRouter()

# --- Helper: Dự đoán cảm xúc (Giống session_manager) ---


def _predict_emotion_single(face_img_bgr):
    if ai_engine.emotion_model is None:
        return "unknown"
    try:
        # Transform cho ResNet
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

        img_rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        input_tensor = tf(pil_img).unsqueeze(0).to(ai_engine.device)

        with torch.no_grad():
            outputs = ai_engine.emotion_model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            return ai_engine.emotion_classes[predicted.item()]
    except Exception:
        return "error"

# 1. API Test Behavior (YOLO)


@router.post("/behavior")
async def test_behavior(file: UploadFile = File(...), annotate: bool = Query(False, description="Return annotated image")):
    if not ai_engine.behavior_model:
        raise HTTPException(status_code=503, detail="Model Behavior chưa load")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = ai_engine.behavior_model(img, verbose=False)
    detections = []

    # Copy image for annotation (if needed)
    annotated_img = img.copy() if annotate else None

    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = ai_engine.behavior_model.names[cls_id]
            label = f"{label_name} ({conf:.2f})"
            x1, y1, x2, y2 = coords.tolist()
            detections.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "color": "#00ff00"
            })
            if annotated_img is not None:
                cv2.rectangle(annotated_img, (x1, y1),
                              (x2, y2), (0, 255, 0), 2)
                # Label background
                text = label_name
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 4)),
                              (x1 + tw + 6, y1), (0, 255, 0), -1)
                cv2.putText(annotated_img, text, (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    if annotate:
        # Encode annotated image to PNG base64
        _, buf = cv2.imencode('.png', annotated_img)
        b64 = base64.b64encode(buf).decode('utf-8')
        return api_response_data(Result.SUCCESS, reply={
            "detections": detections,
            "image_base64": f"data:image/png;base64,{b64}"
        })

    return api_response_data(Result.SUCCESS, detections)

# 2. API Test Emotion (InsightFace + ResNet)


@router.post("/emotion")
async def test_emotion(file: UploadFile = File(...), annotate: bool = Query(False, description="Return annotated image")):
    if not ai_engine.identity_model:
        raise HTTPException(status_code=503, detail="InsightFace chưa load")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    faces = ai_engine.identity_model.get(img)
    detections = []

    annotated_img = img.copy() if annotate else None
    h, w = img.shape[:2]

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        # Crop & Predict Emotion
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        emotion = "unknown"
        if x2c > x1c and y2c > y1c:
            face_img = img[y1c:y2c, x1c:x2c]
            emotion = _predict_emotion_single(face_img)
        detections.append({
            "label": emotion,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "color": "#ff0000"
        })
        if annotated_img is not None:
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = emotion
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, max(0, y1 - th - 4)),
                          (x1 + tw + 6, y1), (255, 0, 0), -1)
            cv2.putText(annotated_img, text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if annotate:
        _, buf = cv2.imencode('.png', annotated_img)
        b64 = base64.b64encode(buf).decode('utf-8')
        return api_response_data(Result.SUCCESS, reply={
            "detections": detections,
            "image_base64": f"data:image/png;base64,{b64}"
        })

    return api_response_data(Result.SUCCESS, detections)
