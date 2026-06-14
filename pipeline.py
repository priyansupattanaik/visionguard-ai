import os
import shutil
from datetime import datetime

import cv2
import numpy as np

from cache_utils import setup_cache
from clip_generator import ClipGenerator
from report_generator import ReportGenerator
from segmenter import GroundedSegmenter
from tracker import ObjectTracker
from vlm import SearchEncoder

setup_cache()


class VisionGuardPipeline:
    def __init__(self, out_dir="output", yolo="yolo11n.pt", clip_model="google/siglip2-base-patch16-224", gdino="IDEA-Research/grounding-dino-tiny", sam="facebook/sam2.1-hiera-tiny"):
        self.out_dir = out_dir
        self.trk = ObjectTracker(model=yolo)
        self.enc = SearchEncoder(model=clip_model)
        self.seg = GroundedSegmenter(gdino=gdino, sam=sam)
        self.idx = None
        self.run_dir = None
        self.clip = None
        self.rep = None
        self.last_hits = []
        os.makedirs(out_dir, exist_ok=True)

    def _new_run(self, video):
        name = os.path.splitext(os.path.basename(video))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.out_dir, f"{name}_{stamp}")
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
        for x in ["frames", "clips", "reports", "segments"]:
            os.makedirs(os.path.join(self.run_dir, x), exist_ok=True)
        self.clip = ClipGenerator(os.path.join(self.run_dir, "clips"))
        self.rep = ReportGenerator(os.path.join(self.run_dir, "reports"))

    def _cos(self, a, b):
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if den == 0 else float(np.dot(a, b) / den)

    def _preview(self, frame, tracks, ts):
        out = frame.copy()
        for t in tracks[:12]:
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 160), 2)
            cv2.putText(out, t["name"], (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 160), 2, cv2.LINE_AA)
        cv2.putText(out, f"{ts:.1f}s", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def index_video_iter(self, video, sample_sec=1.5, win_sec=6.0):
        self._new_run(video)
        self.trk.reset()
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = total / fps if fps else 0.0
        step = max(1, int(round(sample_sec * fps)))
        half = max(1, int(round(win_sec / sample_sec / 2)))
        frames = []
        hist = {}
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step != 0:
                i += 1
                continue
            ts = i / fps
            tracks = self.trk.track(frame, cls=None)
            objs = {}
            tids = []
            move = 0
            still = 0
            for t in tracks:
                name = t["name"]
                objs[name] = objs.get(name, 0) + 1
                tids.append(t["id"])
                x1, y1, x2, y2 = t["box"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                old = hist.get(t["id"])
                if old:
                    dt = max(ts - old["ts"], 1e-6)
                    d = ((cx - old["cx"]) ** 2 + (cy - old["cy"]) ** 2) ** 0.5 / dt
                    if d > 45:
                        move += 1
                    else:
                        still += 1
                hist[t["id"]] = {"cx": cx, "cy": cy, "ts": ts}
            meta = {
                "objects": objs,
                "tracks": sorted(set(tids)),
                "moving_tracks": move,
                "still_people": still if objs.get("person", 0) else 0,
                "person": objs.get("person", 0),
            }
            emb = self.enc.embed_frame(frame)
            frame_path = os.path.join(self.run_dir, "frames", f"f_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append({"frame": i, "ts": ts, "emb": emb, "frame_path": frame_path, "meta": meta})
            yield {"kind": "preview", "image": self._preview(frame, tracks, ts), "status": f"scanning {ts:.1f}s / {dur:.1f}s"}
            i += 1
        cap.release()
        segs = []
        for j, item in enumerate(frames):
            lo = max(0, j - half)
            hi = min(len(frames), j + half + 1)
            chunk = frames[lo:hi]
            emb = np.mean([x["emb"] for x in chunk], axis=0).astype(np.float32)
            emb = emb / max(np.linalg.norm(emb), 1e-6)
            objs = {}
            tids = set()
            for x in chunk:
                tids |= set(x["meta"]["tracks"])
                for k, v in x["meta"]["objects"].items():
                    objs[k] = max(objs.get(k, 0), v)
            segs.append({
                "start": chunk[0]["ts"],
                "end": chunk[-1]["ts"],
                "mid": item["ts"],
                "emb": emb,
                "frame_path": item["frame_path"],
                "objects": sorted(objs.keys()),
                "tracks": sorted(tids),
            })
        self.idx = {"video": video, "meta": {"video": video, "fps": fps, "frames": total, "duration": dur, "sample_sec": sample_sec, "win_sec": win_sec, "segments": len(segs)}, "segments": segs}
        path = os.path.join(self.run_dir, "reports", "index.json")
        self.rep.write_json(path, {"meta": self.idx["meta"], "segments": [{"start": x["start"], "end": x["end"], "mid": x["mid"], "frame_path": x["frame_path"], "objects": x["objects"]} for x in segs]})
        yield {"kind": "done", "meta": self.idx["meta"], "index_json": path}

    def search(self, q, top_k=4):
        qv = self.enc.embed_text(q)
        rows = []
        for seg in self.idx["segments"]:
            score = self._cos(qv, seg["emb"])
            rows.append({
                "query": q,
                "score": score,
                "start": seg["start"],
                "end": seg["end"],
                "frame_path": seg["frame_path"],
                "objects": seg["objects"],
                "tracks": seg["tracks"],
                "summary": ", ".join(seg["objects"]) if seg["objects"] else "no tracked objects",
            })
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        out = []
        for row in rows:
            if len(out) >= top_k:
                break
            if any(abs(row["start"] - x["start"]) < 3 for x in out):
                continue
            out.append(row)
        return out

    def segment_hits(self, hits, query):
        out = []
        for i, hit in enumerate(hits, 1):
            raw = self.clip.extract_clip(self.idx["video"], hit["start"], hit["end"], f"match_{i:02d}_raw", pad=1.5)
            seg_dir = os.path.join(self.run_dir, "segments", f"m_{i:02d}")
            os.makedirs(seg_dir, exist_ok=True)
            seg_mp4 = os.path.join(self.run_dir, "clips", f"match_{i:02d}_seg.mp4")
            seg_clip, frames = self.seg.segment_clip(raw, query, seg_mp4, seg_dir, stride=3)
            row = dict(hit)
            row["raw_clip"] = raw
            row["clip"] = seg_clip
            row["frames"] = frames
            row["label"] = f"{i}. {hit['start']:.2f}s - {hit['end']:.2f}s"
            out.append(row)
        self.last_hits = out
        return out

    def export_selected(self, picks):
        rows = [x for x in self.last_hits if x["label"] in picks]
        if not rows:
            return None, None, None
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        js = self.rep.write_json(os.path.join(self.run_dir, "reports", f"selected_{base}.json"), {"hits": rows})
        csv = self.rep.write_csv(os.path.join(self.run_dir, "reports", f"selected_{base}.csv"), rows)
        html = self.rep.write_html(os.path.join(self.run_dir, "reports", f"selected_{base}.html"), {"query": rows[0]["query"], "video": self.idx["video"], "hits": rows})
        zipf = self.rep.write_zip(os.path.join(self.run_dir, "reports", f"selected_{base}.zip"), [x["clip"] for x in rows] + [x["raw_clip"] for x in rows])
        return zipf, html, csv
