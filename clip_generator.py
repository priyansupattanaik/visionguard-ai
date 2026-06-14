import os
import re
import cv2


class ClipGenerator:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _safe(self, txt):
        txt = re.sub(r"[^a-zA-Z0-9_-]+", "_", txt.strip().lower())
        return txt[:60] or "clip"

    def extract_clip(self, video, st, ed, name, pad=2.0):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        st = max(0.0, st - pad)
        ed = min(total / fps if fps else ed, ed + pad)
        s0 = int(st * fps)
        s1 = int(ed * fps)
        path = os.path.join(self.out_dir, f"{self._safe(name)}_{s0}_{s1}.mp4")
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
        i = s0
        while i <= s1:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            i += 1
        cap.release()
        out.release()
        return path

    def extract_many(self, video, hits, pad=2.0):
        out = []
        for i, hit in enumerate(hits, 1):
            q = hit.get("query") or f"hit_{i}"
            name = f"{i:02d}_{q}"
            path = self.extract_clip(video, hit["start"], hit["end"], name, pad=pad)
            x = dict(hit)
            x["clip"] = path
            out.append(x)
        return out
