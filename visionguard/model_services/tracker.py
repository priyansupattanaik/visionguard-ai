import os
from collections import defaultdict

import torch
from visionguard.runtime.cache import resolve_device

os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    os.path.join(os.getcwd(), ".cache", "visionguard", "ultralytics"),
)
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO


class ObjectTracker:
    def __init__(self, model="yolo11m.pt", conf=0.22, imgsz=640, tracker="botsort.yaml", device=None):
        self.model_name = os.getenv("YOLO_MODEL") or model
        self.conf = float(os.getenv("YOLO_CONF") or conf)
        self.imgsz = int(os.getenv("YOLO_IMGSZ") or imgsz)
        self.tracker = os.getenv("YOLO_TRACKER") or tracker
        self.dev = resolve_device(device)
        self.use_half = self.dev == "cuda"
        self.m = None
        self._track_history = defaultdict(list)

    def reset(self):
        self.m = None
        self._track_history = defaultdict(list)

    def _cached_model_path(self):
        if os.path.dirname(self.model_name):
            if os.path.isfile(self.model_name):
                return self.model_name
            raise FileNotFoundError(f"Configured YOLO model was not found: {self.model_name}")
        base = os.getenv("VISION_GUARD_MODEL_DIR", ".models")
        cached = os.path.join(base, self.model_name)
        if os.path.isfile(cached):
            return cached
        raise FileNotFoundError(
            f"YOLO model '{self.model_name}' is missing from '{base}'. "
            "Run scripts/bootstrap_models.py before indexing a video."
        )

    def load(self):
        if self.m is not None:
            return
        model_path = self._cached_model_path()
        self.m = YOLO(model_path)
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

    def _precision_options(self):
        """Return only precision arguments accepted by the installed Ultralytics runtime."""
        return {"quantize": "fp16"} if self.use_half else {}

    def track(self, frame, cls=None):
        self.load()
        res = self.m.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            imgsz=self.imgsz,
            **self._precision_options(),
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
            **self._precision_options(),
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

    def detect_batch(self, frames, cls=None, conf=None):
        self.load()
        if not frames:
            return []
        res = self.m.predict(
            frames,
            verbose=False,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz,
            **self._precision_options(),
            classes=cls,
        )
        batch = []
        for r in res:
            rows = []
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                for box, cf, ci in zip(boxes, confs, clss):
                    rows.append({
                        "box": [round(x, 2) for x in box],
                        "conf": round(float(cf), 4),
                        "cls": int(ci),
                        "name": self.m.names.get(int(ci), str(ci)),
                    })
            batch.append(rows)
        return batch

    def track_frame(self, frame, frame_idx, ts, cls=None):
        """Track a single frame, recording history for later stats computation.

        Returns list of detections with track IDs for this frame.
        """
        dets = self.track(frame, cls=cls)
        for det in dets:
            tid = det["id"]
            self._track_history[tid].append({
                "frame_idx": frame_idx,
                "ts": ts,
                "box": det["box"],
                "conf": det["conf"],
                "cls": det["cls"],
                "name": det["name"],
            })
        return dets

    def compute_track_stats(self, fps=25.0):
        """Compute per-track statistics from accumulated tracking history.

        Returns dict mapping track_id -> stats dict with:
            - class_name, class_id
            - trajectory_length (number of frames seen)
            - dwell_time (seconds between first and last appearance)
            - entry_frame, exit_frame, entry_ts, exit_ts
            - avg_confidence
            - boxes (list of bounding boxes over time)
        """
        stats = {}
        for tid, history in self._track_history.items():
            if not history:
                continue
            history_sorted = sorted(history, key=lambda x: x["frame_idx"])
            confs = [h["conf"] for h in history_sorted]
            entry = history_sorted[0]
            exit_ = history_sorted[-1]
            dwell = exit_["ts"] - entry["ts"] if len(history_sorted) > 1 else 1.0 / fps
            stats[int(tid)] = {
                "track_id": int(tid),
                "class_name": entry["name"],
                "class_id": entry["cls"],
                "trajectory_length": len(history_sorted),
                "dwell_time": round(dwell, 3),
                "entry_frame": entry["frame_idx"],
                "exit_frame": exit_["frame_idx"],
                "entry_ts": round(entry["ts"], 3),
                "exit_ts": round(exit_["ts"], 3),
                "avg_confidence": round(sum(confs) / len(confs), 4),
                "boxes": [h["box"] for h in history_sorted],
            }
        return stats
