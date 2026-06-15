import os

import torch

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.getcwd(), ".yolo"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, model="yolo11m.pt", conf=0.22, imgsz=1280, tracker="botsort.yaml", device=None):
        self.model_name = model
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = None

    def reset(self):
        self.m = None

    def load(self):
        if self.m is not None:
            return
        self.m = YOLO(self.model_name)
        self.m.to(self.dev)

    def class_ids(self, names):
        self.load()
        want = {str(x).strip().lower() for x in names}
        out = []
        for ci, name in self.m.names.items():
            if str(name).strip().lower() in want:
                out.append(int(ci))
        return out

    def names(self):
        self.load()
        return {int(k): str(v) for k, v in self.m.names.items()}

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
                    "name": self.m.names.get(int(ci), str(ci)),
                })
        return out

    def detect(self, frame, cls=None, conf=None):
        self.load()
        res = self.m.predict(
            frame,
            verbose=False,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz,
            classes=cls,
        )
        out = []
        for r in res:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            for box, cf, ci in zip(boxes, confs, clss):
                out.append({
                    "box": [round(x, 2) for x in box],
                    "conf": round(float(cf), 4),
                    "cls": int(ci),
                    "name": self.m.names.get(int(ci), str(ci)),
                })
        return out
