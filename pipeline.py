import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from cache_utils import setup_cache
from clip_generator import ClipGenerator
from report_generator import ReportGenerator
from segmenter import GroundedSegmenter
from tracker import ObjectTracker
from vector_index import SegmentVectorIndex
from vlm import SearchEncoder

setup_cache()


class VisionGuardPipeline:
    def __init__(self, out_dir="output", yolo="yolo11s.pt", clip_model="google/siglip2-base-patch16-224", locate_model="nvidia/LocateAnything-3B", sam="facebook/sam2.1-hiera-small"):
        self.out_dir = out_dir
        self.trk = ObjectTracker(model=yolo)
        self.enc = SearchEncoder(model=clip_model)
        self.seg = GroundedSegmenter(sam=sam, locate_model=locate_model)
        self.idx = None
        self.run_dir = None
        self.clip = None
        self.rep = None
        self.last_hits = []
        self.search_idx = SegmentVectorIndex(bit_width=4)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.raw_jobs = {}
        self.seg_jobs = {}
        os.makedirs(out_dir, exist_ok=True)

    def _clip_name(self, i, kind):
        return f"match_{i:02d}_{kind}"

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1 = max(ax1, bx1)
        y1 = max(ay1, by1)
        x2 = min(ax2, bx2)
        y2 = min(ay2, by2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        bb = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / (aa + bb - inter)

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
        self.raw_jobs = {}
        self.seg_jobs = {}

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

    def _q_objs(self, q):
        q = f" {q.strip().lower()} "
        m = {
            "person": [" person ", " people ", " man ", " woman ", " human "],
            "car": [" car ", " vehicle ", " sedan "],
            "truck": [" truck ", " lorry "],
            "bus": [" bus "],
            "motorcycle": [" motorcycle ", " motorbike ", " bike ", " scooter "],
            "bicycle": [" bicycle ", " cycle "],
            "backpack": [" backpack ", " bag ", " parcel ", " package "],
            "suitcase": [" suitcase ", " luggage ", " parcel ", " package "],
            "handbag": [" handbag ", " purse ", " parcel ", " package "],
        }
        out = set()
        for k, rows in m.items():
            if any(x in q for x in rows):
                out.add(k)
        return sorted(out)

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
        block = max(1, int(round(win_sec / sample_sec)))
        segs = []
        for j, item in enumerate(frames):
            lo = (j // block) * block
            hi = min(len(frames), lo + block)
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
                "seg_id": np.uint64(len(segs)),
                "start": chunk[0]["ts"],
                "end": chunk[-1]["ts"],
                "mid": item["ts"],
                "emb": emb,
                "frame_path": item["frame_path"],
                "objects": sorted(objs.keys()),
                "tracks": sorted(tids),
                "tags": [],
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
        seg_vecs = np.ascontiguousarray(np.stack([x["emb"] for x in segs]).astype(np.float32)) if segs else np.zeros((0, 0), dtype=np.float32)
        seg_ids = np.asarray([x["seg_id"] for x in segs], dtype=np.uint64) if segs else np.zeros((0,), dtype=np.uint64)
        self.search_idx.build(seg_vecs, seg_ids, path=os.path.join(self.run_dir, "reports", "segment_index.tvim"))
        path = os.path.join(self.run_dir, "reports", "index.json")
        self.rep.write_json(
            path,
            {
                "meta": {**self.idx["meta"], "retriever": self.search_idx.backend},
                "segments": [
                    {
                        "seg_id": int(x["seg_id"]),
                        "start": x["start"],
                        "end": x["end"],
                        "mid": x["mid"],
                        "frame_path": x["frame_path"],
                        "objects": x["objects"],
                        "tags": x["tags"],
                    }
                    for x in segs
                ],
            },
        )
        yield {"kind": "done", "meta": {**self.idx["meta"], "retriever": self.search_idx.backend}, "index_json": path}

    def search(self, q, top_k=4):
        qv = self.enc.embed_text(q)
        ql = q.strip().lower()
        qobjs = self._q_objs(q)
        n = len(self.idx["segments"])
        if n == 0:
            return []
        seg_map = {int(x["seg_id"]): x for x in self.idx["segments"]}
        fetch_k = min(max(top_k * 8, 24), n)
        base_scores, seg_ids = self.search_idx.search(qv, fetch_k)
        rows = []
        for base_score, seg_id in zip(base_scores, seg_ids):
            seg = seg_map.get(int(seg_id))
            if seg is None:
                continue
            score = float(base_score)
            sobj = set(seg["objects"])
            if qobjs:
                hit = len(sobj & set(qobjs))
                if hit:
                    score += 0.12 * hit
                else:
                    score -= 0.12
            if "sitting" in ql and "person" in sobj:
                score += 0.05
            rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "start": seg["start"],
                "end": seg["end"],
                "frame_path": seg["frame_path"],
                "objects": seg["objects"],
                "tracks": seg["tracks"],
                "tags": seg["tags"],
                "summary": ", ".join(seg["objects"]) if seg["objects"] else "no tracked objects",
            })
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        out = []
        for row in rows:
            if len(out) >= top_k:
                break
            if any(abs(row["start"] - x["start"]) < 3 for x in out):
                continue
            if row["score"] < 0.18:
                continue
            out.append(row)
        return out

    def prepare_hits(self, hits, query):
        out = []
        for i, hit in enumerate(hits, 1):
            row = dict(hit)
            row["match_id"] = i
            row["raw_clip"] = None
            row["clip"] = None
            row["frames"] = []
            row["segmented"] = False
            row["label"] = f"{i}. {hit['start']:.2f}s - {hit['end']:.2f}s"
            out.append(row)
        self.last_hits = out
        if out:
            self._ensure_raw_clip(out[0], wait=True)
            self._start_segment(out[0], query)
            for row in out[1:]:
                self._ensure_raw_clip(row, wait=False)
        return out

    def _build_raw_clip(self, row):
        name = self._clip_name(row["match_id"], "raw")
        path = self.clip.clip_path(self.idx["video"], row["start"], row["end"], name, pad=1.5)
        if os.path.exists(path):
            return path
        return self.clip.extract_clip(self.idx["video"], row["start"], row["end"], name, pad=1.5)

    def _ensure_raw_clip(self, row, wait=True):
        if row["raw_clip"]:
            return row["raw_clip"]
        job = self.raw_jobs.get(row["match_id"])
        if job is None:
            if wait:
                row["raw_clip"] = self._build_raw_clip(row)
                row["clip"] = row["raw_clip"]
                return row["raw_clip"]
            self.raw_jobs[row["match_id"]] = self.pool.submit(self._build_raw_clip, dict(row))
            return None
        if not wait and not job.done():
            return None
        row["raw_clip"] = job.result()
        if not row["clip"]:
            row["clip"] = row["raw_clip"]
        return row["raw_clip"]

    def _segment_payload(self, row, query):
        raw = self._build_raw_clip(row)
        seg_dir = os.path.join(self.run_dir, "segments", f"m_{row['match_id']:02d}")
        os.makedirs(seg_dir, exist_ok=True)
        seg_mp4 = os.path.join(self.run_dir, "clips", f"{self._clip_name(row['match_id'], 'seg')}.mp4")
        seg_clip, frames, seen = self.seg.segment_clip(raw, query, seg_mp4, seg_dir, stride=3)
        return {"raw_clip": raw, "clip": seg_clip if seen > 0 else raw, "frames": frames, "seen": seen}

    def _start_segment(self, row, query):
        if row["segmented"] or row["match_id"] in self.seg_jobs:
            return
        self.seg_jobs[row["match_id"]] = self.pool.submit(self._segment_payload, dict(row), query)

    def _collect_segment(self, row):
        job = self.seg_jobs.get(row["match_id"])
        if job is None or not job.done():
            return False
        payload = job.result()
        row["raw_clip"] = payload["raw_clip"]
        row["clip"] = payload["clip"]
        row["frames"] = payload["frames"]
        row["segmented"] = bool(payload["seen"] > 0)
        if payload["seen"] == 0 and "no grounded mask, showing raw clip" not in row["summary"]:
            row["summary"] = f"{row['summary']} | no grounded mask, showing raw clip"
        return True

    def _ensure_segment(self, row, query):
        if row["segmented"]:
            return row
        job = self.seg_jobs.get(row["match_id"])
        if job is None:
            payload = self._segment_payload(row, query)
        else:
            payload = job.result()
        row["raw_clip"] = payload["raw_clip"]
        row["clip"] = payload["clip"]
        row["frames"] = payload["frames"]
        row["segmented"] = bool(payload["seen"] > 0)
        if payload["seen"] == 0 and "no grounded mask, showing raw clip" not in row["summary"]:
            row["summary"] = f"{row['summary']} | no grounded mask, showing raw clip"
        return row

    def _segment_row(self, row, query):
        return self._ensure_segment(row, query)

    def pick_match(self, label, query):
        for x in self.last_hits:
            if x["label"] == label:
                self._ensure_raw_clip(x, wait=True)
                ready = self._collect_segment(x)
                if not ready:
                    self._start_segment(x, query)
                gal = [(fp, x["label"]) for fp in x["frames"]]
                clip = x["clip"] if x["clip"] else x["raw_clip"]
                txt = f"### {x['label']}\n\n{x['summary']}"
                if not ready and not x["frames"]:
                    txt = f"{txt}\n\nsegmentation is preparing in the background. Open this match again in a few seconds."
                return clip, gal, txt
        return None, [], ""

    def export_selected(self, picks, query):
        rows = [x for x in self.last_hits if x["label"] in picks]
        if not rows:
            return None, None, None
        for row in rows:
            self._ensure_segment(row, query)
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        js = self.rep.write_json(os.path.join(self.run_dir, "reports", f"selected_{base}.json"), {"hits": rows})
        csv = self.rep.write_csv(os.path.join(self.run_dir, "reports", f"selected_{base}.csv"), rows)
        html = self.rep.write_html(os.path.join(self.run_dir, "reports", f"selected_{base}.html"), {"query": rows[0]["query"], "video": self.idx["video"], "hits": rows})
        zipf = self.rep.write_zip(os.path.join(self.run_dir, "reports", f"selected_{base}.zip"), [x["clip"] for x in rows] + [x["raw_clip"] for x in rows])
        return zipf, html, csv
