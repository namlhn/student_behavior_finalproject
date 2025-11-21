import os
import torch
from ultralytics import YOLO
from torchvision import models
from pathlib import Path


class AIEngine:
    def __init__(self):
        print("Loading AI Engine...")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Load YOLOv8 (Behavior)
        base_dir = Path(__file__).resolve().parents[1]
        models_dir = base_dir / "models"
        yolo_path = str(models_dir / "yolov8_best.pt")
        self.behavior_model = None
        try:
            if os.path.exists(yolo_path):
                self.behavior_model = YOLO(yolo_path)
                print("YOLOv8 model loaded.")
            else:
                print(
                    f"[WARN] YOLO weights not found at {yolo_path}. Behavior detection disabled.")
        except Exception as e:
            print(
                f"[WARN] Failed to load YOLOv8: {e}. Behavior detection disabled.")

        # 2. Load InsightFace (Identity) - optional
        self.identity_model = None
        try:
            import insightface  # lazy optional import
            provider = 'CUDAExecutionProvider' if self.device.type == 'cuda' else 'CPUExecutionProvider'
            self.identity_model = insightface.app.FaceAnalysis(providers=[
                                                               provider])
            self.identity_model.prepare(
                ctx_id=0 if self.device.type == 'cuda' else -1, det_size=(640, 640))
            print("InsightFace model loaded.")
        except Exception as e:
            print(
                f"[WARN] Failed to load InsightFace: {e}. Identity recognition disabled.")

        # 3. Load Emotion Model (ResNet18)
        emotion_path = str(models_dir / "best_resnet18_sgd.ckpt")
        self.emotion_model = None
        try:
            if os.path.exists(emotion_path):
                self.emotion_model = models.resnet18(weights=None)
                self.emotion_model.fc = torch.nn.Linear(
                    self.emotion_model.fc.in_features, 7)
                ckpt = torch.load(emotion_path, map_location=self.device)
                state = ckpt.get("model", ckpt)
                self.emotion_model.load_state_dict(state)
                self.emotion_model.to(self.device)
                self.emotion_model.eval()
                print("Emotion model loaded.")
            else:
                print(
                    f"[WARN] Emotion model checkpoint not found at {emotion_path}. Emotion classification disabled.")
        except Exception as e:
            print(
                f"[WARN] Failed to load Emotion model: {e}. Emotion classification disabled.")

        self.emotion_transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
        self.emotion_classes = ["angry", "disgust",
                                "fear", "happy", "neutral", "sad", "surprise"]


ai_engine = AIEngine()
