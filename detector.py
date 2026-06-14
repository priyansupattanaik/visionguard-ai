"""
Object Detection Module using YOLO11m.
Handles frame-by-frame detection with confidence filtering.
"""
import cv2
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import numpy as np


class ObjectDetector:
    def __init__(self, model_path: str = "yolo11m.pt", conf_threshold: float = 0.45, device: str = None):
        """
        Initialize YOLO11m detector.
        
        Args:
            model_path: Path to YOLO11m weights (auto-downloads if not present)
            conf_threshold: Minimum confidence for detection
            device: 'cuda', 'cpu', or None (auto)
        """
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[Detector] Loading YOLO11m on {self.device}...")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"[Detector] Model loaded. Classes: {self.model.names}")
        
    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Dict]:
        """
        Run detection on a single frame.
        
        Returns:
            List of detections: [{bbox: [x1,y1,x2,y2], conf: float, class_id: int, class_name: str}]
        """
        results = self.model(frame, verbose=False, conf=self.conf_threshold, classes=classes)
        detections = []
        
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": round(conf, 3),
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id]
                })
        return detections
    
    def detect_batch(self, frames: List[np.ndarray], classes: Optional[List[int]] = None) -> List[List[Dict]]:
        """Batch detection for faster processing."""
        results = self.model(frames, verbose=False, conf=self.conf_threshold, classes=classes)
        all_detections = []
        
        for r in results:
            frame_dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    frame_dets.append({
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": round(float(box.conf[0].cpu().numpy()), 3),
                        "class_id": int(box.cls[0].cpu().numpy()),
                        "class_name": self.model.names[int(box.cls[0].cpu().numpy())]
                    })
            all_detections.append(frame_dets)
        return all_detections