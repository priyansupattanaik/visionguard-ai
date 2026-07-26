import cv2
import numpy as np


class DecordVideoReader:
    def __init__(self, path):
        self.path = path
        try:
            from decord import VideoReader, cpu
            self.vr = VideoReader(path, ctx=cpu(0))
            self.fps = float(self.vr.get_avg_fps() or 25.0)
            self.count = len(self.vr)
        except Exception as exc:
            raise ValueError(f"Decord could not decode video: {path}") from exc
        first = self.get_frame(0) if self.count else None
        if first is None:
            self.width = 0
            self.height = 0
        else:
            self.height, self.width = first.shape[:2]

    def __len__(self):
        return self.count

    def _to_bgr(self, frame):
        if frame is None:
            return None
        if hasattr(frame, "asnumpy"):
            frame = frame.asnumpy()
        if frame is None or frame.size == 0:
            return None
        return cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)

    def get_frame(self, idx):
        if idx < 0 or idx >= self.count:
            return None
        try:
            return self._to_bgr(self.vr[idx])
        except Exception:
            return None

    def get_batch(self, indices):
        picked = [int(x) for x in indices if 0 <= int(x) < self.count]
        if not picked:
            return []
        try:
            batch = self.vr.get_batch(picked).asnumpy()
            return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in batch]
        except Exception as exc:
            raise RuntimeError("Decord failed while decoding a requested frame batch.") from exc

    def ts_for(self, idx):
        return float(idx) / self.fps if self.fps else 0.0
