"""
Multi-Object Tracking using ByteTrack (via Ultralytics).
Maintains consistent IDs across frames for behavior analysis.
"""
from typing import List, Dict, Tuple
from ultralytics import YOLO
import numpy as np


class ObjectTracker:
    def __init__(self, model_path: str = "yolo11m.pt", conf_threshold: float = 0.45, 
                 track_buffer: int = 30, match_thresh: float = 0.8):
        """
        Initialize ByteTrack tracker.
        
        Args:
            track_buffer: Frames to keep lost tracks alive
            match_thresh: Matching threshold for track association
        """
        self.conf_threshold = conf_threshold
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        
        # ByteTrack is integrated in ultralytics; we use model.track()
        self.model = YOLO(model_path)
        print("[Tracker] ByteTrack initialized via YOLO11m")
        
    def track(self, frame: np.ndarray, persist: bool = True, 
              classes: List[int] = None) -> List[Dict]:
        """
        Track objects in frame.
        
        Returns:
            List of tracked objects: [{track_id, bbox, conf, class_id, class_name}]
        """
        results = self.model.track(
            frame, 
            persist=persist,
            verbose=False,
            conf=self.conf_threshold,
            tracker="bytetrack.yaml",
            classes=classes
        )
        
        tracks = []
        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue
                
            for box, track_id in zip(r.boxes, r.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                tracks.append({
                    "track_id": int(track_id.cpu().numpy()),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": round(float(box.conf[0].cpu().numpy()), 3),
                    "class_id": int(box.cls[0].cpu().numpy()),
                    "class_name": self.model.names[int(box.cls[0].cpu().numpy())]
                })
        return tracks