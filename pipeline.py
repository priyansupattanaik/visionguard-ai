import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np

from cache_utils import setup_cache
from clip_generator import ClipGenerator
from florence import FlorenceVerifier
from report_generator import ReportGenerator
from segmenter import GroundedSegmenter
from tracker import ObjectTracker
from vector_index import SegmentVectorIndex
from vlm import SearchEncoder

setup_cache()


class VisionGuardPipeline:
    def __init__(self, out_dir="output", yolo="yolo11m.pt", clip_model="google/siglip2-so400m-patch14-384", florence_model="microsoft/Florence-2-large", sam="facebook/sam2.1-hiera-small"):
        self.out_dir = out_dir
        self.trk = ObjectTracker(model=yolo)
        self.enc = SearchEncoder(model=clip_model)
        self.ver = FlorenceVerifier(model=florence_model)
        self.seg = GroundedSegmenter(sam=sam, florence_model=florence_model, florence=self.ver)
        self.idx = None
        self.run_dir = None
        self.clip = None
        self.rep = None
        self.last_hits = []
        self.search_idx = SegmentVectorIndex(bit_width=4)
        self.frame_idx = SegmentVectorIndex(bit_width=4)
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.raw_jobs = {}
        self.seg_jobs = {}
        os.makedirs(out_dir, exist_ok=True)

    def _color_words(self):
        return {
            "yellow": np.array([220.0, 190.0, 60.0], dtype=np.float32),
            "white": np.array([215.0, 215.0, 215.0], dtype=np.float32),
            "black": np.array([35.0, 35.0, 35.0], dtype=np.float32),
            "gray": np.array([135.0, 135.0, 135.0], dtype=np.float32),
            "red": np.array([180.0, 65.0, 65.0], dtype=np.float32),
            "blue": np.array([70.0, 110.0, 185.0], dtype=np.float32),
            "green": np.array([80.0, 150.0, 90.0], dtype=np.float32),
            "orange": np.array([210.0, 140.0, 65.0], dtype=np.float32),
            "brown": np.array([125.0, 95.0, 70.0], dtype=np.float32),
        }

    def _query_colors(self, q):
        q = f" {self._normalize_query(q)} "
        return [x for x in self._color_words().keys() if f" {x} " in q]

    def _estimate_color(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(w - 1, x1))
        x2 = max(1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(1, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).reshape(-1, 3).astype(np.float32)
        bright = rgb.mean(axis=1)
        sat = rgb.max(axis=1) - rgb.min(axis=1)
        keep = sat > 18
        if keep.any():
            rgb = rgb[keep]
        mean = rgb.mean(axis=0)
        palette = self._color_words()
        best = None
        best_dist = 1e9
        for name, ref in palette.items():
            d = float(np.linalg.norm(mean - ref))
            if d < best_dist:
                best = name
                best_dist = d
        if bright.mean() > 205 and sat.mean() < 30:
            return "white"
        if bright.mean() < 55:
            return "black"
        if sat.mean() < 22:
            return "gray"
        return best

    def _appearance_tags(self, frame, tracks):
        tags = []
        for t in tracks:
            name = t["name"]
            if name not in {"car", "truck", "bus", "motorcycle", "bicycle"}:
                continue
            color = self._estimate_color(frame, t["box"])
            if color:
                tags.append(f"{color} {name}")
            tags.append(name)
        return sorted(set(tags))

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
        q = f" {self._normalize_query(q)} "
        m = {
            "person": [" person ", " people ", " peoples ", " persons ", " man ", " woman ", " human "],
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
        if any(x in q for x in [" accident ", " collision ", " crash ", " hit-and-run ", " pileup "]):
            out |= {"car", "truck", "bus", "motorcycle", "bicycle"}
        return sorted(out)

    def _normalize_query(self, q):
        q = q.strip().lower()
        repl = {
            "peoples": "people",
            "persons": "person",
            "human beings": "people",
            "bike accident": "motorcycle accident",
            "bikes": "bicycle",
            "cycles": "bicycle",
            "cars": "car",
            "trucks": "truck",
            "buses": "bus",
        }
        for src, dst in repl.items():
            q = q.replace(src, dst)
        return " ".join(q.split())

    def _query_variants(self, q):
        ql = self._normalize_query(q)
        out = [ql]
        groups = {
            "accident": [
                "traffic accident",
                "vehicle collision",
                "car crash",
                "vehicles hitting each other",
            ],
            "collision": [
                "traffic collision",
                "vehicle crash",
                "cars colliding",
            ],
            "crash": [
                "vehicle crash",
                "traffic accident",
                "cars hitting each other",
            ],
            "fight": [
                "people fighting",
                "physical fight",
                "person attacking another person",
            ],
            "fall": [
                "person falling",
                "person on the ground after a fall",
                "human fall incident",
            ],
            "crowd": [
                "crowd of people",
                "many people gathered together",
                "group of people",
            ],
            "loitering": [
                "person standing around",
                "person waiting near one place",
                "person staying in the same area",
            ],
        }
        for key, vals in groups.items():
            if key in ql:
                out.extend(vals)
        seen = set()
        uniq = []
        for item in out:
            if item not in seen:
                uniq.append(item)
                seen.add(item)
        return uniq

    def _embed_query(self, q):
        vecs = [self.enc.embed_text(x) for x in self._query_variants(q)]
        mix = np.mean(vecs, axis=0).astype(np.float32)
        den = max(np.linalg.norm(mix), 1e-6)
        return mix / den

    def _frame_summary(self, q, peak_ts, objs):
        label = ", ".join(objs) if objs else "no tracked objects"
        return f"best matching sampled frame at {peak_ts:.2f}s | detected: {label}"

    def _clip_bounds(self, ts, pad=None):
        pad = self.idx["meta"]["sample_sec"] if pad is None else pad
        dur = self.idx["meta"]["duration"]
        return max(0.0, ts - pad), min(dur, ts + pad)

    def _verify_rows(self, rows, query, top_n=6):
        if not rows:
            return rows
        q_text_vec = self.enc.embed_text(query)
        take = min(top_n, len(rows))
        for i in range(take):
            caption = self.ver.caption_path(rows[i]["frame_path"])
            rows[i]["verified_caption"] = caption
            if not caption:
                rows[i]["verify_score"] = 0.0
                continue
            c_vec = self.enc.embed_text(caption)
            vscore = self._cos(q_text_vec, c_vec)
            rows[i]["verify_score"] = float(vscore)
            rows[i]["score"] = float(rows[i]["score"] * 0.75 + max(vscore, 0.0) * 0.25)
            rows[i]["summary"] = f"best matching sampled frame at {rows[i].get('peak_ts', rows[i]['start']):.2f}s | {caption}"
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        return rows

    def _cluster_frame_hits(self, rows, top_k, gap_sec):
        rows = sorted(rows, key=lambda x: x["ts"])
        clusters = []
        for row in rows:
            if not clusters or row["ts"] - clusters[-1][-1]["ts"] > gap_sec:
                clusters.append([row])
            else:
                clusters[-1].append(row)
        out = []
        for chunk in clusters:
            peak = max(chunk, key=lambda x: x["score"])
            objs = sorted({obj for row in chunk for obj in row["objects"]})
            start, end = self._clip_bounds(peak["ts"], pad=max(gap_sec, self.idx["meta"]["sample_sec"]))
            out.append({
                "query": peak["query"],
                "score": max(x["score"] for x in chunk),
                "base_score": peak["base_score"],
                "start": start,
                "end": end,
                "peak_ts": peak["ts"],
                "frame_path": peak["frame_path"],
                "objects": objs,
                "tracks": sorted({tid for row in chunk for tid in row["tracks"]}),
                "appearances": sorted({tag for row in chunk for tag in row.get("appearances", [])}),
                "tags": [],
                "summary": self._frame_summary(peak["query"], peak["ts"], objs),
            })
        out = sorted(out, key=lambda x: x["score"], reverse=True)
        dedup = []
        for row in out:
            if len(dedup) >= top_k:
                break
            if any(abs(row["peak_ts"] - x["peak_ts"]) < gap_sec for x in dedup):
                continue
            dedup.append(row)
        return dedup

    def _fallback_object_hits(self, q, top_k):
        qobjs = set(self._q_objs(q))
        qcolors = set(self._query_colors(q))
        if not qobjs:
            return []
        rows = []
        for row in self.idx.get("frames", []):
            sobj = set(row["objects"])
            hit = len(sobj & qobjs)
            if not hit:
                continue
            appear = set(row.get("appearances", []))
            color_hit = 0
            if qcolors:
                for color in qcolors:
                    for obj in qobjs:
                        if f"{color} {obj}" in appear:
                            color_hit += 1
                if color_hit == 0:
                    continue
            score = 0.2 + 0.08 * hit + 0.14 * color_hit
            rows.append({
                "query": q,
                "score": score,
                "base_score": score,
                "ts": row["ts"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
            })
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        if not rows:
            return []
        hits = []
        for row in rows:
            if len(hits) >= top_k:
                break
            if any(abs(row["ts"] - x["peak_ts"]) < 2.0 for x in hits):
                continue
            start, end = self._clip_bounds(row["ts"])
            hits.append({
                "query": q,
                "score": row["score"],
                "base_score": row["base_score"],
                "start": start,
                "end": end,
                "peak_ts": row["ts"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
                "tags": [],
                "summary": self._frame_summary(q, row["ts"], row["objects"]),
            })
        for hit in hits:
            appear = ", ".join(hit.get("appearances", []))
            suffix = f" | appearance: {appear}" if appear else ""
            hit["summary"] = f"object-matched sampled frame at {hit['peak_ts']:.2f}s | detected: {', '.join(hit['objects'])}{suffix}"
            hit["low_confidence"] = True
        return hits

    def index_video_iter(self, video, sample_sec=0.75, win_sec=4.5):
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
                "appearances": self._appearance_tags(frame, tracks),
                "still_people": still if objs.get("person", 0) else 0,
                "person": objs.get("person", 0),
            }
            emb = self.enc.embed_frame(frame)
            frame_path = os.path.join(self.run_dir, "frames", f"f_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append({
                "frame_id": np.uint64(len(frames)),
                "frame": i,
                "ts": ts,
                "emb": emb,
                "frame_path": frame_path,
                "meta": meta,
            })
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
        self.idx = {
            "video": video,
            "meta": meta,
            "frames": [
                {
                    "frame_id": int(x["frame_id"]),
                    "frame": x["frame"],
                    "ts": x["ts"],
                    "frame_path": x["frame_path"],
                    "objects": sorted(x["meta"]["objects"].keys()),
                    "appearances": x["meta"]["appearances"],
                    "tracks": x["meta"]["tracks"],
                }
                for x in frames
            ],
            "segments": segs,
        }
        frame_vecs = np.ascontiguousarray(np.stack([x["emb"] for x in frames]).astype(np.float32)) if frames else np.zeros((0, 0), dtype=np.float32)
        frame_ids = np.asarray([x["frame_id"] for x in frames], dtype=np.uint64) if frames else np.zeros((0,), dtype=np.uint64)
        self.frame_idx.build(frame_vecs, frame_ids, path=os.path.join(self.run_dir, "reports", "frame_index.tvim"))
        seg_vecs = np.ascontiguousarray(np.stack([x["emb"] for x in segs]).astype(np.float32)) if segs else np.zeros((0, 0), dtype=np.float32)
        seg_ids = np.asarray([x["seg_id"] for x in segs], dtype=np.uint64) if segs else np.zeros((0,), dtype=np.uint64)
        self.search_idx.build(seg_vecs, seg_ids, path=os.path.join(self.run_dir, "reports", "segment_index.tvim"))
        path = os.path.join(self.run_dir, "reports", "index.json")
        self.rep.write_json(
            path,
            {
                "meta": {
                    **self.idx["meta"],
                    "retriever": self.frame_idx.backend,
                    "segment_retriever": self.search_idx.backend,
                    "verifier": self.ver.model_name,
                },
                "frames": [
                    {
                        "frame_id": x["frame_id"],
                        "ts": x["ts"],
                        "frame_path": x["frame_path"],
                        "objects": x["objects"],
                        "appearances": x["appearances"],
                    }
                    for x in self.idx["frames"]
                ],
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
        yield {
            "kind": "done",
                "meta": {
                    **self.idx["meta"],
                    "retriever": self.frame_idx.backend,
                    "segment_retriever": self.search_idx.backend,
                    "verifier": self.ver.model_name,
                },
            "index_json": path,
        }

    def search(self, q, top_k=4):
        q = self._normalize_query(q)
        qv = self._embed_query(q)
        ql = q
        qobjs = self._q_objs(q)
        qcolors = set(self._query_colors(q))
        frames = self.idx.get("frames", [])
        frame_map = {int(x["frame_id"]): x for x in frames}
        fetch_k = min(max(top_k * 12, 36), len(frames))
        frame_scores, frame_ids = self.frame_idx.search(qv, fetch_k)
        rows = []
        for base_score, frame_id in zip(frame_scores, frame_ids):
            row = frame_map.get(int(frame_id))
            if row is None:
                continue
            score = float(base_score)
            sobj = set(row["objects"])
            appear = set(row.get("appearances", []))
            if qobjs:
                hit = len(sobj & set(qobjs))
                if hit:
                    score += 0.1 * hit
                else:
                    score -= 0.08
            if qcolors and qobjs:
                color_hit = 0
                for color in qcolors:
                    for obj in qobjs:
                        if f"{color} {obj}" in appear:
                            color_hit += 1
                if color_hit:
                    score += 0.22 * color_hit
                else:
                    score -= 0.12
            if any(x in ql for x in ["accident", "collision", "crash"]) and sobj & {"car", "truck", "bus", "motorcycle", "bicycle"}:
                score += 0.08
            if "sitting" in ql and "person" in sobj:
                score += 0.05
            rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "ts": row["ts"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "appearances": row.get("appearances", []),
                "tracks": row["tracks"],
            })
        ranked_rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        rows = [x for x in ranked_rows if x["score"] >= 0.14]
        out = self._cluster_frame_hits(rows, top_k=top_k, gap_sec=max(self.idx["meta"]["sample_sec"] * 1.25, 1.0))
        if out:
            return self._verify_rows(out, q, top_n=max(top_k * 2, 4))[:top_k]
        obj_hits = self._fallback_object_hits(q, top_k)
        if obj_hits:
            return obj_hits
        if ranked_rows:
            weak = self._cluster_frame_hits(ranked_rows[: max(top_k * 3, 8)], top_k=top_k, gap_sec=max(self.idx["meta"]["sample_sec"] * 1.25, 1.0))
            for hit in weak:
                hit["summary"] = f"low-confidence visual match at {hit['peak_ts']:.2f}s | detected: {', '.join(hit['objects']) if hit['objects'] else 'no tracked objects'}"
                hit["low_confidence"] = True
            if weak:
                return weak
        n = len(self.idx["segments"])
        if n == 0:
            return []
        seg_map = {int(x["seg_id"]): x for x in self.idx["segments"]}
        fetch_k = min(max(top_k * 8, 24), n)
        base_scores, seg_ids = self.search_idx.search(qv, fetch_k)
        seg_rows = []
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
                    score -= 0.1
            if any(x in ql for x in ["accident", "collision", "crash"]) and sobj & {"car", "truck", "bus", "motorcycle", "bicycle"}:
                score += 0.08
            if "sitting" in ql and "person" in sobj:
                score += 0.05
            seg_rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "start": seg["start"],
                "end": seg["end"],
                "peak_ts": seg["mid"],
                "frame_path": seg["frame_path"],
                "objects": seg["objects"],
                "tracks": seg["tracks"],
                "tags": seg["tags"],
                "summary": self._frame_summary(q, seg["mid"], seg["objects"]),
            })
        seg_rows = sorted(seg_rows, key=lambda x: x["score"], reverse=True)
        out = []
        for row in seg_rows:
            if len(out) >= top_k:
                break
            if any(abs(row["peak_ts"] - x["peak_ts"]) < 3 for x in out):
                continue
            if row["score"] < 0.18:
                continue
            out.append(row)
        return self._verify_rows(out, q, top_n=max(top_k * 2, 4))[:top_k]

    def prepare_hits(self, hits, query):
        out = []
        for i, hit in enumerate(hits, 1):
            row = dict(hit)
            row["match_id"] = i
            row["raw_clip"] = None
            row["clip"] = None
            row["frames"] = []
            row["segmented"] = False
            row["label"] = f"{i}. {hit.get('peak_ts', hit['start']):.2f}s"
            out.append(row)
        self.last_hits = out
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
