import os
import shutil
from datetime import datetime

import cv2
import numpy as np

from clip_generator import ClipGenerator
from report_generator import ReportGenerator
from tracker import ObjectTracker
from vlm import QwenVerifier, SearchEncoder


class VisionGuardPipeline:
    def __init__(self, out_dir="output", yolo="yolo11n.pt", clip_model="google/siglip2-base-patch16-224", qwen_model="Qwen/Qwen2.5-VL-7B-Instruct"):
        self.out_dir = out_dir
        self.trk = ObjectTracker(model=yolo)
        self.enc = SearchEncoder(model=clip_model)
        self.vfy = QwenVerifier(model=qwen_model)
        self.idx = None
        self.run_dir = None
        self.clip = None
        self.rep = None
        os.makedirs(out_dir, exist_ok=True)
        self.obj_keys = {
            "person": ["person", "people", "man", "woman", "boy", "girl", "human"],
            "car": ["car", "sedan", "hatchback"],
            "truck": ["truck", "lorry"],
            "bus": ["bus"],
            "motorcycle": ["motorcycle", "bike", "biker"],
            "bicycle": ["bicycle", "cycle"],
            "dog": ["dog"],
            "cat": ["cat"],
        }
        self.act_keys = {
            "still": ["standing", "sitting", "idle", "waiting", "loitering"],
            "move": ["walking", "running", "moving", "entering", "leaving", "crossing"],
            "crowd": ["crowd", "group", "many people"],
        }

    def _new_run(self, video):
        name = os.path.splitext(os.path.basename(video))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.out_dir, f"{name}_{stamp}")
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
        os.makedirs(os.path.join(self.run_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "reports"), exist_ok=True)
        self.clip = ClipGenerator(os.path.join(self.run_dir, "clips"))
        self.rep = ReportGenerator(os.path.join(self.run_dir, "reports"))

    def _cos(self, a, b):
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den == 0:
            return 0.0
        return float(np.dot(a, b) / den)

    def _obj_bonus(self, q, objs):
        q = q.lower()
        names = set(objs.keys())
        hit = 0.0
        for name, keys in self.obj_keys.items():
            if any(k in q for k in keys) and name in names:
                hit += 0.12
        return min(hit, 0.36)

    def _act_bonus(self, q, meta):
        q = q.lower()
        hit = 0.0
        for k in self.act_keys["still"]:
            if k in q and meta.get("still_people", 0) > 0:
                hit += 0.12
                break
        for k in self.act_keys["move"]:
            if k in q and meta.get("moving_tracks", 0) > 0:
                hit += 0.12
                break
        for k in self.act_keys["crowd"]:
            if k in q and meta.get("person", 0) >= 3:
                hit += 0.12
                break
        return min(hit, 0.24)

    def _track_meta(self, tracks, hist, ts):
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
        return {
            "objects": objs,
            "tracks": sorted(set(tids)),
            "moving_tracks": move,
            "still_people": still if objs.get("person", 0) else 0,
            "person": objs.get("person", 0),
        }

    def _meta_txt(self, meta):
        obj = ", ".join(f"{k}:{v}" for k, v in sorted(meta["objects"].items()))
        return f"objects[{obj}] moving[{meta['moving_tracks']}] still_people[{meta['still_people']}] tracks[{len(meta['tracks'])}]"

    def _save_frame(self, frame, i):
        path = os.path.join(self.run_dir, "frames", f"f_{i:06d}.jpg")
        cv2.imwrite(path, frame)
        return path

    def _merge_hits(self, hits, gap=2.0):
        if not hits:
            return []
        hits = sorted(hits, key=lambda x: x["start"])
        out = [dict(hits[0])]
        for cur in hits[1:]:
            prv = out[-1]
            if cur["start"] <= prv["end"] + gap:
                prv["end"] = max(prv["end"], cur["end"])
                prv["score"] = max(prv["score"], cur["score"])
                prv["objects"] = sorted(set(prv["objects"]) | set(cur["objects"]))
                prv["tracks"] = sorted(set(prv["tracks"]) | set(cur["tracks"]))
                prv["summary"] = cur["summary"] if cur["score"] >= prv["score"] else prv["summary"]
                prv["meta_txt"] = cur["meta_txt"] if cur["score"] >= prv["score"] else prv["meta_txt"]
                continue
            out.append(dict(cur))
        return out

    def _seg_summary(self, meta):
        objs = meta["objects"]
        if not objs:
            return "no tracked objects"
        parts = [f"{v} {k}" for k, v in sorted(objs.items(), key=lambda x: (-x[1], x[0]))]
        tag = []
        if meta.get("still_people", 0) > 0:
            tag.append("still person")
        if meta.get("moving_tracks", 0) > 0:
            tag.append("moving track")
        if meta.get("person", 0) >= 3:
            tag.append("crowd")
        return ", ".join(parts + tag[:2])

    def index_video(self, video, cls=None, sample_sec=1.0, win_sec=6.0, progress=None):
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
            tracks = self.trk.track(frame, cls=cls)
            meta = self._track_meta(tracks, hist, ts)
            emb = self.enc.embed_frame(frame)
            frame_path = self._save_frame(frame, i)
            frames.append({
                "frame": i,
                "ts": ts,
                "emb": emb,
                "frame_path": frame_path,
                "meta": meta,
            })
            if progress and total:
                progress(min(ts / max(dur, 1e-6), 0.98), desc=f"indexing {ts:0.1f}s / {dur:0.1f}s")
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
            moving = 0
            still = 0
            for x in chunk:
                moving += x["meta"]["moving_tracks"]
                still += x["meta"]["still_people"]
                tids |= set(x["meta"]["tracks"])
                for k, v in x["meta"]["objects"].items():
                    objs[k] = max(objs.get(k, 0), v)
            meta = {
                "objects": objs,
                "tracks": sorted(tids),
                "moving_tracks": moving,
                "still_people": still,
                "person": objs.get("person", 0),
            }
            segs.append({
                "start": chunk[0]["ts"],
                "end": chunk[-1]["ts"],
                "mid": item["ts"],
                "emb": emb,
                "frame_path": item["frame_path"],
                "meta": meta,
                "summary": self._seg_summary(meta),
                "meta_txt": self._meta_txt(meta),
            })
        meta = {
            "video": video,
            "fps": fps,
            "frames": total,
            "duration": dur,
            "sample_sec": sample_sec,
            "win_sec": win_sec,
            "segments": len(segs),
        }
        self.idx = {"video": video, "meta": meta, "segments": segs}
        slim = {
            "meta": meta,
            "segments": [{
                "start": x["start"],
                "end": x["end"],
                "mid": x["mid"],
                "frame_path": x["frame_path"],
                "meta": x["meta"],
                "summary": x["summary"],
            } for x in segs]
        }
        path = os.path.join(self.run_dir, "reports", "index.json")
        self.rep.write_json(path, slim)
        return {"run_dir": self.run_dir, "meta": meta, "index_json": path}

    def search(self, q, top_k=6):
        if not self.idx:
            raise ValueError("index not ready")
        qv = self.enc.embed_text(q)
        rows = []
        for seg in self.idx["segments"]:
            base = self._cos(qv, seg["emb"])
            bonus = self._obj_bonus(q, seg["meta"]["objects"]) + self._act_bonus(q, seg["meta"])
            rows.append({
                "query": q,
                "score": base + bonus,
                "start": seg["start"],
                "end": seg["end"],
                "summary": seg["summary"],
                "objects": sorted(seg["meta"]["objects"].keys()),
                "tracks": seg["meta"]["tracks"],
                "frame_path": seg["frame_path"],
                "meta_txt": seg["meta_txt"],
            })
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        rows = self._merge_hits(rows[: max(top_k * 3, 12)])
        return rows[:top_k]

    def verify(self, q, hits, clip_pad=2.0, progress=None):
        if not self.idx:
            raise ValueError("index not ready")
        out = []
        n = max(len(hits), 1)
        for i, hit in enumerate(hits, 1):
            clip = self.clip.extract_clip(self.idx["video"], hit["start"], hit["end"], f"qwen_{i:02d}", pad=clip_pad)
            chk = self.vfy.verify(clip, q, meta=hit["meta_txt"])
            row = dict(hit)
            row["clip"] = clip
            row["match"] = chk["match"]
            row["qwen_score"] = round(chk["score"], 4)
            row["summary"] = chk["summary"] or row["summary"]
            out.append(row)
            if progress:
                progress(i / n, desc=f"qwen verify {i}/{n}")
        out.sort(key=lambda x: (x["match"], x["qwen_score"], x["score"]), reverse=True)
        return out

    def export_hits(self, q, hits):
        if not self.idx:
            raise ValueError("index not ready")
        payload = {
            "query": q,
            "video": self.idx["video"],
            "meta": self.idx["meta"],
            "hits": hits,
        }
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        js = self.rep.write_json(os.path.join(self.run_dir, "reports", f"hits_{base}.json"), payload)
        csv = self.rep.write_csv(os.path.join(self.run_dir, "reports", f"hits_{base}.csv"), hits)
        html = self.rep.write_html(os.path.join(self.run_dir, "reports", f"hits_{base}.html"), payload)
        zipf = self.rep.write_zip(os.path.join(self.run_dir, "reports", f"clips_{base}.zip"), [x["clip"] for x in hits])
        return {"hits": hits, "json": js, "csv": csv, "html": html, "zip": zipf}
