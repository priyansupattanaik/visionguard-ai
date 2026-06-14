import os

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".yolo"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model="yolo11n.pt", conf=0.35, imgsz=960, tracker="bytetrack.yaml", device=None):
        self.model_name = model
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = None
        self.names = {}

    def load(self):
        if self.m is not None:
            return
        self.m = YOLO(self.model_name)
        self.m.to(self.dev)
        self.names = self.m.names

    def reset(self):
        self.m = None
        self.names = {}

    def track(self, frame, cls=None):
        self.load()
        res = self.m.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            classes=cls,
        )
        out = []
        for r in res:
            if r.boxes is None or r.boxes.id is None:
                continue
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            for tid, box, cf, ci in zip(ids, boxes, confs, clss):
                out.append({
                    "id": int(tid),
                    "box": [round(x, 2) for x in box],
                    "conf": round(float(cf), 4),
                    "cls": int(ci),
                    "name": self.names.get(int(ci), str(ci)),
                })
        return out
