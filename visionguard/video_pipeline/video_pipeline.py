import os
import json
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime

import cv2
import numpy as np

from visionguard.runtime.cache import setup_cache
from visionguard.runtime.settings import PipelineSettings
from visionguard.model_services.clip_generator import ClipGenerator
from visionguard.model_services.nvidia_verifier import NvidiaFrameVerifier
from visionguard.model_services.model_provider import (
    ModelProviderError,
    create_model_provider,
    model_health_snapshot,
)
from visionguard.model_services.report_generator import ReportGenerator
from visionguard.model_services.segmenter import GroundedSegmenter
from visionguard.model_services.tracker import ObjectTracker
from visionguard.video_pipeline.vector_index import SegmentVectorIndex, _as_2d_float32
from visionguard.video_pipeline.video_reader import DecordVideoReader
from visionguard.video_pipeline.detector_evidence import DetectorEvidenceRetriever
from visionguard.video_pipeline.frame_dedup import is_exact_duplicate
from visionguard.model_services.vlm import SearchEncoder
from visionguard.search import DeterministicQueryPlanner, VideoQueryGraph
from visionguard.semantic import EventExtractor, NvidiaSemanticAnalyzer, SemanticAnalysisError

setup_cache()
logger = logging.getLogger(__name__)


def _stack_embeddings(vectors):
    """Stack 1D embeddings into a contiguous 2D float32 array (N, D)."""
    return _as_2d_float32(vectors)


class VisionGuardPipeline:
    def __init__(self, out_dir="output", yolo="yolo11m.pt", clip_model="google/siglip2-so400m-patch14-384", verifier_model=None, sam="facebook/sam2.1-hiera-small"):
        self.settings = PipelineSettings.from_env(out_dir)
        self.out_dir = self.settings.out_dir
        self.trk = ObjectTracker(model=os.getenv("YOLO_MODEL") or yolo)
        self.enc = SearchEncoder(model=os.getenv("CLIP_MODEL") or clip_model)
        self.query_planner = DeterministicQueryPlanner()
        self.query_graph = VideoQueryGraph(
            self.query_planner,
            self._graph_object_evidence,
            self._graph_event_evidence,
            self._graph_zone_evidence,
            self._graph_semantic_evidence,
            verify=self._graph_verify_evidence,
            retrieve_count=self._graph_count_evidence,
            retrieve_temporal=self._graph_temporal_evidence,
        )
        self.model_provider = create_model_provider()
        self._model_health_cache = (0.0, None)
        self.vlm = self.enc
        self.ver = NvidiaFrameVerifier(model=os.getenv("NVIDIA_VLM_MODEL") or verifier_model)
        self.seg = GroundedSegmenter(sam=os.getenv("SAM_MODEL") or sam, verifier_model=os.getenv("VERIFIER_MODEL") or verifier_model, verifier=self.ver)
        self.idx = None
        self.run_dir = None
        self.clip = None
        self.rep = None
        self.last_hits = []
        self.search_idx = SegmentVectorIndex(bit_width=4)
        self.frame_idx = SegmentVectorIndex(bit_width=4)
        self.crop_idx = SegmentVectorIndex(bit_width=4)
        self.crop_meta = []
        self.track_stats = {}
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.verifier_ready_timeout = self.settings.verifier_ready_timeout
        self.verifier_poll_interval = self.settings.verifier_poll_interval
        self.raw_jobs = {}
        self.seg_jobs = {}
        os.makedirs(self.out_dir, exist_ok=True)
        self._warmup_failures = {}
        self.zero_query = None
        self.last_query_plan = None
        self.last_query_message = ""
        self._warmup_done = False
        self.semantic_events = []

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
        ch, cw = crop.shape[:2]
        mx1 = int(cw * 0.15)
        mx2 = int(cw * 0.85)
        my1 = int(ch * 0.15)
        my2 = int(ch * 0.85)
        core = crop[my1:my2, mx1:mx2] if mx2 > mx1 and my2 > my1 else crop
        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        hh = hsv[..., 0].astype(np.float32)
        ss = hsv[..., 1].astype(np.float32)
        vv = hsv[..., 2].astype(np.float32)
        valid = vv > 40
        if not valid.any():
            return None
        sat_valid = valid & (ss > 45)
        if sat_valid.any():
            hue = hh[sat_valid]
            blue_ratio = float(((hue >= 95) & (hue <= 130)).mean())
            red_ratio = float(((hue <= 10) | (hue >= 170)).mean())
            green_ratio = float(((hue >= 35) & (hue <= 90)).mean())
            yellow_ratio = float(((hue >= 18) & (hue <= 35)).mean())
            orange_ratio = float(((hue >= 10) & (hue < 18)).mean())
            if blue_ratio >= 0.28:
                return "blue"
            if red_ratio >= 0.28:
                return "red"
            if green_ratio >= 0.28:
                return "green"
            if yellow_ratio >= 0.24:
                return "yellow"
            if orange_ratio >= 0.22:
                return "orange"
        bright = vv[valid]
        sat = ss[valid]
        if bright.mean() > 205 and sat.mean() < 32:
            return "white"
        if bright.mean() < 55:
            return "black"
        if sat.mean() < 24:
            return "gray"
        return None

    def _appearance_tags(self, frame, detections):
        tags = []
        for t in detections:
            name = t["name"]
            color = self._estimate_color(frame, t["box"])
            if color:
                tags.append(f"{color} {name}")
            tags.append(name)
        return sorted(set(tags))

    def _clip_name(self, i, kind):
        return f"match_{i:02d}_{kind}"

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

    def _preview(self, frame, tracks, ts):
        out = frame.copy()
        for t in tracks[:12]:
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 160), 2)
            cv2.putText(out, t["name"], (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 160), 2, cv2.LINE_AA)
        cv2.putText(out, f"{ts:.1f}s", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    def _q_objs(self, q):
        try:
            detector_labels = self.trk.names().values()
        except Exception:
            detector_labels = ()
        return self.query_planner.resolve_entities(self._normalize_query(q), detector_labels)

    def _is_temporal_query(self, q):
        """Check if query uses controlled temporal terms that can be answered via tracks."""
        q = f" {self._normalize_query(q)} "
        temporal_terms = {
            " loitering ", " loiter ", " suspicious ",
            " enter ", " entering ", " entry ",
            " exit ", " exiting ", " leave ", " leaving ",
            " approach ", " approaching ",
            " gather ", " gathering ", " crowd ", " crowded ",
        }
        return any(term in q for term in temporal_terms)

    def _normalize_query(self, q):
        return " ".join(re.findall(r"[a-z0-9]+", q.strip().casefold()))

    def _query_detector_classes(self, q):
        qobjs = self._q_objs(q)
        if not qobjs:
            return [], {}
        class_ids = self.trk.class_ids(qobjs)
        if not class_ids:
            return [], {}
        names = self.trk.names()
        want = {str(x).strip().lower() for x in qobjs}
        cls_to_name = {int(ci): str(names.get(int(ci), ci)).strip().lower() for ci in class_ids if str(names.get(int(ci), ci)).strip().lower() in want}
        return class_ids, cls_to_name

    def _is_strict_object_query(self, q):
        try:
            labels = self.trk.names().values()
        except Exception:
            labels = ()
        plan = self.query_planner.plan(q, detector_labels=labels)
        return bool(plan.entities) and not plan.unknown_terms and not plan.events

    def _matching_detections(self, row, qobjs, qcolors, cls_to_name=None):
        out = []
        wanted = {str(x).strip().lower() for x in qobjs}
        cls_to_name = cls_to_name or {}
        for det in row.get("detections", []):
            name = str(det.get("name", "")).strip().lower()
            if wanted and name not in wanted:
                continue
            color = det.get("color")
            if qcolors:
                if not color or color not in qcolors:
                    continue
            if cls_to_name and int(det.get("cls", -1)) in cls_to_name:
                name = cls_to_name[int(det["cls"])]
            out.append({
                "name": name,
                "box": det["box"],
                "conf": det.get("conf", 0.0),
                "cls": det.get("cls"),
                "color": color,
            })
        return out

    def _refine_detector_hits(self, q, top_k):
        """Aggregate calibrated detector evidence into temporal segments.

        A high-confidence frame is evidence, not a complete event. Grouping
        nearby matched frames gives retrieval a defensible start/end interval
        while retaining the strongest source frame for inspection.
        """
        class_ids, cls_to_name = self._query_detector_classes(q)
        if not class_ids:
            return []
        qobjs = set(self._q_objs(q))
        qcolors = set(self._query_colors(q))
        retriever = DetectorEvidenceRetriever(
            self._matching_detections,
            self._clip_bounds,
            self.settings.minimum_evidence_confidence,
        )
        return retriever.retrieve(self.idx, q, qobjs, qcolors, class_ids, cls_to_name, top_k)


    def _draw_boxes(self, src_path, boxes, out_name, label_text=None):
        if not src_path or not os.path.exists(src_path):
            return src_path
        frame = cv2.imread(src_path)
        if frame is None:
            return src_path
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 120), 2)
        if label_text:
            cv2.putText(frame, label_text, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out_path = os.path.join(self.run_dir, "frames", out_name)
        cv2.imwrite(out_path, frame)
        return out_path

    def _attach_gallery_frame(self, row, query):
        src = row.get("representative_frame_path") or row.get("frame_path")
        if not src:
            row["gallery_frame"] = src
            return row
        boxes = row.get("det_boxes", [])
        if not boxes:
            boxes = [item.get("box") for item in row.get("matched_detections", []) if item.get("box")]
        if boxes:
            stamp = int(round(row.get("peak_ts", row.get("start", 0.0)) * 100))
            row["gallery_frame"] = self._draw_boxes(src, boxes, f"gallery_{row.get('match_id', 0):02d}_{stamp:06d}.jpg", label_text=f"{query} @ {row.get('peak_ts', row.get('start', 0.0)):.2f}s")
        else:
            row["gallery_frame"] = src
        return row

    def _query_variants(self, q):
        original = " ".join(q.strip().lower().split())
        ql = self._normalize_query(q)
        out = [original, ql] if original and original != ql else [ql]
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
        return f"best matching indexed frame at {peak_ts:.2f}s | detected: {label}"

    def _clip_bounds(self, ts, pad=None):
        pad = self.idx["meta"]["frame_interval_sec"] if pad is None else pad
        dur = self.idx["meta"]["duration"]
        return max(0.0, ts - pad), min(dur, ts + pad)

    def _reselect_best_frame(self, video_path, start_sec, end_sec, query_vec, step_sec=0.1):
        vr = DecordVideoReader(video_path)
        if len(vr) == 0:
            return None, start_sec, -1.0
        step_frames = max(1, int(round(step_sec * vr.fps)))
        start_idx = max(0, int(round(start_sec * vr.fps)))
        end_idx = min(len(vr) - 1, int(round(end_sec * vr.fps)))
        best_score = -1.0
        best_frame = None
        best_ts = start_sec
        indices = list(range(start_idx, end_idx + 1, step_frames))
        chunk_size = 16
        for offset in range(0, len(indices), chunk_size):
            chunk = indices[offset: offset + chunk_size]
            frames = vr.get_batch(chunk)
            for idx, frame in zip(chunk, frames):
                if frame is None:
                    continue
                emb = self.vlm.embed_frame(frame)
                score = float(np.dot(emb, query_vec))
                ts = vr.ts_for(idx)
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
                    best_ts = ts
        if best_frame is None:
            return None, start_sec, -1.0
        safe_start = str(round(start_sec, 2)).replace(".", "_")
        out_path = os.path.join(self.run_dir, "frames", f"resel_{safe_start}.jpg")
        cv2.imwrite(out_path, best_frame)
        return out_path, best_ts, best_score

    def _refresh_det_boxes_for_hit(self, hit, query):
        class_ids, _ = self._query_detector_classes(query)
        if not class_ids:
            return hit.get("det_boxes", [])
        frame_path = hit.get("representative_frame_path") or hit.get("frame_path")
        if not frame_path or not os.path.exists(frame_path):
            return hit.get("det_boxes", [])
        frame = cv2.imread(frame_path)
        if frame is None:
            return hit.get("det_boxes", [])
        want = set(self._q_objs(query))
        dets = self.trk.detect(frame, cls=class_ids, conf=0.12)
        boxes = [det["box"] for det in dets if str(det.get("name", "")).strip().lower() in want]
        return boxes or hit.get("det_boxes", [])

    def _apply_reselection(self, hits, query, query_vec, top_n=1):
        if self.embedding_mode() == "metadata_embeddings":
            return hits
        take = min(top_n, len(hits))
        for i in range(take):
            frame_path, best_ts, best_score = self._reselect_best_frame(self.idx["video"], hits[i]["start"], hits[i]["end"], query_vec, step_sec=0.1)
            if not frame_path:
                continue
            hits[i]["representative_frame_path"] = frame_path
            hits[i]["frame_path"] = frame_path
            hits[i]["peak_ts"] = best_ts
            hits[i]["reselected_score"] = best_score
            hits[i]["det_boxes"] = self._refresh_det_boxes_for_hit(hits[i], query)
            if hits[i].get("matched_detections"):
                labels = sorted({x["name"] for x in hits[i]["matched_detections"]})
            hits[i]["summary"] = f"detector-matched indexed frame at {best_ts:.2f}s | detected: {', '.join(labels)}"
        return hits

    def _mark_unverified(self, rows, mode=None):
        mode = mode or self.verification_mode()
        for row in rows:
            row["verified_match"] = False
            row["verify_score"] = 0.0
            row["verified_caption"] = ""
            row["grounded"] = False
            row["verification_mode"] = mode
            row["low_confidence"] = True
            if not row.get("summary"):
                row["summary"] = f"detector/retrieval candidate (verification: {mode})"
        return rows

    def _apply_verify_result(self, row, result, query):
        boxes = result.get("boxes", [])
        caption = result.get("caption", "")
        matched = bool(result.get("matched"))
        confidence = float(result.get("confidence", 0.0) or 0.0)
        verification_mode = result.get("verification_mode") or self.verification_mode()
        row["det_boxes"] = boxes or row.get("det_boxes", [])
        row["verified_caption"] = caption
        row["grounded"] = bool(boxes)
        row["verified_match"] = matched
        row["verify_score"] = confidence
        row["verification_mode"] = verification_mode
        if matched:
            row["score"] = float(row["score"] + min(0.35, 0.16 + 0.18 * confidence))
            label = ", ".join(row.get("objects", [])) or query
            detail = caption or f"visible match for {query}"
            if verification_mode == "nvidia_api":
                row["summary"] = f"NVIDIA-verified query match at {row.get('peak_ts', row['start']):.2f}s | {detail} | detected: {label}"
            elif verification_mode == "llama_cpp_vision":
                row["summary"] = f"Local vision-verified query match at {row.get('peak_ts', row['start']):.2f}s | {detail} | detected: {label}"
            else:
                row["summary"] = f"unconfirmed query match at {row.get('peak_ts', row['start']):.2f}s | {detail} | detected: {label}"
        elif caption:
            row["score"] = float(row["score"] * 0.6)
            row["low_confidence"] = True
            row["summary"] = f"unverified visual candidate at {row.get('peak_ts', row['start']):.2f}s | {caption}"
        else:
            row["score"] = float(row["score"] * 0.5)
            row["low_confidence"] = True
            if verification_mode == "unknown" or verification_mode.endswith(("_unconfigured", "_unavailable", "_disabled")):
                row["summary"] = (
                    f"detector/retrieval candidate at {row.get('peak_ts', row['start']):.2f}s "
                    f"(verification unavailable: {verification_mode})"
                )
        return row

    def _verify_rows(self, rows, query, top_n=1):
        if not rows:
            return rows
        mode = self.verification_mode()
        if not self.ver.is_ready():
            self._mark_unverified(rows, mode=mode)
            return sorted(rows, key=lambda x: x["score"], reverse=True)
        take = min(top_n, len(rows))
        futures = {
            self.pool.submit(
                self.ver.verify_query,
                rows[i].get("representative_frame_path", rows[i]["frame_path"]),
                query,
                rows[i].get("cache_key"),
            ): i
            for i in range(take)
        }
        results = [None] * take
        for future, idx in futures.items():
            try:
                results[idx] = future.result(timeout=30)
            except Exception:
                results[idx] = {
                    "matched": False,
                    "confidence": 0.0,
                    "caption": "",
                    "boxes": [],
                    "verification_mode": self.verification_mode(),
                }
        for i, result in enumerate(results):
            if result is None:
                result = {
                    "matched": False,
                    "confidence": 0.0,
                    "caption": "",
                    "boxes": [],
                    "verification_mode": self.verification_mode(),
                }
            self._apply_verify_result(rows[i], result, query)
        for j in range(take, len(rows)):
            rows[j]["verification_mode"] = rows[j].get("verification_mode") or self.verification_mode()
            rows[j]["low_confidence"] = True
        rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        return rows

    def _verify_rows_stream(self, rows, query, top_n=1):
        if not rows:
            return
        mode = self.verification_mode()
        if not self.ver.is_ready():
            self._mark_unverified(rows, mode=mode)
            for i, row in enumerate(rows[:top_n]):
                yield i, row
            return
        take = min(top_n, len(rows))
        futures = {
            self.pool.submit(
                self.ver.verify_query,
                rows[i].get("representative_frame_path", rows[i]["frame_path"]),
                query,
                rows[i].get("cache_key"),
            ): i
            for i in range(take)
        }
        results = [None] * take
        for future, idx in futures.items():
            try:
                results[idx] = future.result(timeout=30)
            except Exception:
                results[idx] = {
                    "matched": False,
                    "confidence": 0.0,
                    "caption": "",
                    "boxes": [],
                    "verification_mode": self.verification_mode(),
                }
        for i, result in enumerate(results):
            if result is None:
                result = {
                    "matched": False,
                    "confidence": 0.0,
                    "caption": "",
                    "boxes": [],
                    "verification_mode": self.verification_mode(),
                }
            self._apply_verify_result(rows[i], result, query)
            yield i, rows[i]

    def _confirmed_rows(self, rows):
        return [x for x in rows if x.get("verified_match")]

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
            start, end = self._clip_bounds(peak["ts"], pad=max(gap_sec, self.idx["meta"]["frame_interval_sec"]))
            out.append({
                "query": peak["query"],
                "score": max(x["score"] for x in chunk),
                "base_score": peak["base_score"],
                "cache_key": f"frame:{peak.get('frame_id', peak['ts'])}",
                "start": start,
                "end": end,
                "peak_ts": peak["ts"],
                "representative_frame_path": peak["frame_path"],
                "frame_path": peak["frame_path"],
                "objects": objs,
                "tracks": sorted({tid for row in chunk for tid in row["tracks"]}),
                "appearances": sorted({tag for row in chunk for tag in row.get("appearances", [])}),
                "det_boxes": [x["box"] for x in peak.get("detections", [])],
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
                "retrieval_mode": "object_fallback",
                "frame_id": row.get("frame_id"),
                "ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
                "det_boxes": [x["box"] for x in row.get("detections", [])],
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
                "retrieval_mode": "object_fallback",
                "cache_key": f"frame:{row.get('frame_id', row['ts'])}",
                "start": start,
                "end": end,
                "peak_ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "tracks": row["tracks"],
                "appearances": row.get("appearances", []),
                "det_boxes": row.get("det_boxes", []),
                "tags": [],
                "summary": self._frame_summary(q, row["ts"], row["objects"]),
            })
        for hit in hits:
            appear = ", ".join(hit.get("appearances", []))
            suffix = f" | appearance: {appear}" if appear else ""
            hit["summary"] = f"object-matched indexed frame at {hit['peak_ts']:.2f}s | detected: {', '.join(hit['objects'])}{suffix}"
            hit["low_confidence"] = True
        return hits

    def index_video_iter(self, video, win_sec=None):
        if win_sec is None:
            win_sec = self.settings.win_sec
        enable_crop_embeddings = self.settings.enable_crop_embeddings
        self._new_run(video)
        self.trk.reset()
        vr = DecordVideoReader(video)
        if len(vr) == 0:
            raise ValueError(f"cannot open video: {video}")
        fps = vr.fps or 25.0
        total = len(vr)
        dur = total / fps if fps else 0.0
        frame_interval_sec = 1.0 / fps
        frames = []
        pending = []
        # Align decode batch with embedding batch; keep a small floor for IO efficiency.
        batch_size = max(4, min(16, int(self.enc.image_batch_size)))
        frame_vec_chunks = []
        frame_id_chunks = []
        crop_vec_chunks = []
        crop_id_chunks = []
        crop_meta_list = []
        self.crop_meta = []
        self.crop_idx = SegmentVectorIndex(bit_width=4)
        previous_frame = None
        last_kept_objects = set()
        t0 = time.perf_counter()
        timings = {
            "frame_read_filter": 0.0,
            "detection_tracking": 0.0,
            "frame_embeddings": 0.0,
            "crop_embeddings": 0.0,
            "index_building": 0.0,
        }
        processed_samples = 0
        kept_frames = 0
        skipped_duplicates = 0
        total_frames = total
        logger.info(
            "scan_start video=%s frames=%s fps=%s deduplication=exact_consecutive_pixels win_sec=%s device=%s yolo=%s image_batch=%s crop_embeddings=%s",
            os.path.basename(video), total, fps, win_sec, self.enc.dev,
            self.trk.model_name, self.enc.image_batch_size, enable_crop_embeddings,
        )

        def flush_pending():
            nonlocal frames, pending, frame_vec_chunks, frame_id_chunks, crop_vec_chunks, crop_id_chunks, crop_meta_list
            if not pending:
                return
            write_futures = [item["write_future"] for item in pending]
            embedding_started = time.perf_counter()
            emb_list = self.enc.embed_frames(
                [x["frame"] for x in pending],
                metadata=[x["meta"] for x in pending],
            )
            timings["frame_embeddings"] += time.perf_counter() - embedding_started
            for future in write_futures:
                if not future.result():
                    raise RuntimeError("Failed to write an extracted evidence frame.")
            chunk_vecs = []
            chunk_ids = []
            for item, emb in zip(pending, emb_list):
                frame_id = np.uint64(len(frames))
                frames.append({
                    "frame_id": frame_id,
                    "frame": item["frame_idx"],
                    "ts": item["ts"],
                    "emb": emb,
                    "frame_path": item["frame_path"],
                    "meta": item["meta"],
                })
                chunk_vecs.append(emb)
                chunk_ids.append(frame_id)
            if chunk_vecs:
                frame_vec_chunks.append(_stack_embeddings(chunk_vecs))
                frame_id_chunks.append(np.asarray(chunk_ids, dtype=np.uint64))

            # Crop embeddings are optional because each detection adds another
            # expensive SigLIP inference workload.
            if not enable_crop_embeddings:
                pending = []
                return
            crops = []
            c_info = []
            for item in pending:
                frame = item["frame"]
                for det in item["meta"]["detections"]:
                    x1, y1, x2, y2 = [int(round(v)) for v in det["box"]]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crops.append(crop)
                            c_info.append({
                                "frame_idx": item["frame_idx"],
                                "ts": item["ts"],
                                "track_id": det.get("track_id"),
                                "name": det["name"],
                                "box": det["box"]
                            })
            if crops:
                # Batch embed crops in chunks to avoid OOM
                crop_embedding_started = time.perf_counter()
                c_embs = self.enc.embed_frames(
                    crops,
                    metadata=[{"objects": {row["name"]: 1}, "detections": [row]} for row in c_info],
                )
                timings["crop_embeddings"] += time.perf_counter() - crop_embedding_started
                batch_crop_vecs = []
                batch_crop_ids = []
                for emb, info in zip(c_embs, c_info):
                    cid = np.uint64(len(crop_meta_list))
                    batch_crop_vecs.append(emb)
                    batch_crop_ids.append(cid)
                    crop_meta_list.append(info)
                if batch_crop_vecs:
                    crop_vec_chunks.append(_stack_embeddings(batch_crop_vecs))
                    crop_id_chunks.append(np.asarray(batch_crop_ids, dtype=np.uint64))

            pending = []

        for offset in range(0, total, batch_size):
            read_filter_started = time.perf_counter()
            chunk = list(range(offset, min(total, offset + batch_size)))
            batch_frames = vr.get_batch(chunk)
            if len(batch_frames) != len(chunk) or any(frame is None for frame in batch_frames):
                raise RuntimeError(f"Decoder did not return every requested frame in range {chunk[0]}-{chunk[-1]}.")
            valid = list(zip(chunk, batch_frames))
            processed_before_batch = processed_samples
            processed_samples += len(chunk)
            retained = []
            for i, frame in valid:
                ts = vr.ts_for(i)
                if is_exact_duplicate(previous_frame, frame):
                    skipped_duplicates += 1
                    continue
                retained.append((i, frame, ts))
                previous_frame = frame.copy()
            timings["frame_read_filter"] += time.perf_counter() - read_filter_started
            if not retained:
                continue
            # Sequential tracking preserves BoT-SORT state across frames.
            for (i, frame, ts) in retained:
                detection_started = time.perf_counter()
                detections = self.trk.track_frame(frame, frame_idx=i, ts=ts, cls=None)
                track_ids = [det["id"] for det in detections if "id" in det]
                timings["detection_tracking"] += time.perf_counter() - detection_started
                objs = {}
                det_rows = []
                for det in detections:
                    name = det["name"]
                    objs[name] = objs.get(name, 0) + 1
                    color = None
                    color = self._estimate_color(frame, det["box"])
                    det_row = {
                        "box": det["box"],
                        "conf": det["conf"],
                        "cls": det["cls"],
                        "name": name,
                        "color": color,
                    }
                    if "track_id" in det:
                        det_row["track_id"] = int(det["track_id"])
                    elif "id" in det:
                        det_row["track_id"] = int(det["id"])
                    det_rows.append(det_row)
                meta = {
                    "objects": objs,
                    "tracks": sorted(set(int(x) for x in track_ids)),
                    "appearances": self._appearance_tags(frame, det_rows),
                    "detections": det_rows,
                    "deduplication": "unique",
                    "object_delta": len(set(objs.keys()) ^ last_kept_objects),
                }
                frame_path = os.path.join(self.run_dir, "frames", f"f_{i:06d}.jpg")
                write_future = self.pool.submit(cv2.imwrite, frame_path, frame)
                pending.append({
                    "frame_idx": i,
                    "ts": ts,
                    "frame": frame.copy(),
                    "frame_path": frame_path,
                    "meta": meta,
                    "write_future": write_future,
                })
                kept_frames += 1
                last_kept_objects = set(objs.keys())
                current_processed = processed_before_batch + chunk.index(i) + 1
                elapsed = max(1e-6, time.perf_counter() - t0)
                sample_rate = current_processed / elapsed
                remain = max(0, total_frames - current_processed)
                eta = remain / sample_rate if sample_rate > 0 else 0.0
                pct = min(100.0, 100.0 * current_processed / total_frames)
                status = (
                    f"scanning {ts:.1f}s / {dur:.1f}s | {pct:.0f}% | eta {eta:.1f}s | "
                    f"indexed {kept_frames} | exact duplicates {skipped_duplicates} | det {timings['detection_tracking']:.0f}s | "
                    f"emb {timings['frame_embeddings']:.0f}s"
                )
                yield {
                    "kind": "preview",
                    "image": self._preview(frame, det_rows, ts),
                    "status": status,
                    "frame_number": int(i),
                    "timestamp_ms": int(i * 1000.0 / fps),
                    "processed_samples": int(current_processed),
                    "total_samples": int(total_frames),
                    "kept_frames": int(kept_frames),
                    "detections": int(len(det_rows)),
                    "frame_path": frame_path,
                    "objects": sorted(meta["objects"].keys()),
                    "tracks": meta["tracks"],
                    "detection_rows": det_rows,
                    "selection_reason": "unique",
                }
                if len(pending) >= self.enc.image_batch_size:
                    flush_pending()
        flush_pending()
        # Compute track statistics from accumulated tracking data
        track_stats = self.trk.compute_track_stats(fps=fps)
        self.track_stats = track_stats
        segs = []
        seg_vec_chunks = []
        seg_id_chunks = []
        seg_chunk_vecs = []
        seg_chunk_ids = []
        for chunk in self._group_frames_by_time(frames, win_sec):
            emb = np.mean([np.asarray(x["emb"], dtype=np.float32).reshape(-1) for x in chunk], axis=0).astype(np.float32)
            emb = emb / max(float(np.linalg.norm(emb)), 1e-6)
            emb = emb.astype(np.float32)
            objs = {}
            tids = set()
            object_delta = 0
            for x in chunk:
                tids |= set(x["meta"]["tracks"])
                for k, v in x["meta"]["objects"].items():
                    objs[k] = max(objs.get(k, 0), v)
                object_delta += int(x["meta"].get("object_delta", 0))
            segs.append({
                "seg_id": np.uint64(len(segs)),
                "start": chunk[0]["ts"],
                "end": chunk[-1]["ts"],
                "mid": chunk[len(chunk) // 2]["ts"],
                "emb": emb,
                "frame_path": chunk[len(chunk) // 2]["frame_path"],
                "objects": sorted(objs.keys()),
                "tracks": sorted(tids),
                "temporal_stats": {
                    "object_delta_sum": object_delta,
                },
                "tags": [],
            })
            seg_chunk_vecs.append(emb)
            seg_chunk_ids.append(np.uint64(len(segs) - 1))
        if seg_chunk_vecs:
            seg_vec_chunks.append(_stack_embeddings(seg_chunk_vecs))
            seg_id_chunks.append(np.asarray(seg_chunk_ids, dtype=np.uint64))
        meta = {
            "video": video,
            "fps": fps,
            "frames": total,
            "duration": dur,
            "frame_interval_sec": frame_interval_sec,
            "win_sec": win_sec,
            "segments": len(segs),
            "decoded_frames": total_frames,
            "kept_frames": kept_frames,
            "skipped_exact_duplicate_frames": skipped_duplicates,
            "enable_crop_embeddings": enable_crop_embeddings,
            "device": self.enc.dev,
            "embedding_mode": self.enc.mode(),
            "embedding_dimension": int(frame_vec_chunks[0].shape[1]) if frame_vec_chunks else 0,
            "nonzero_frame_vectors": int(sum(np.count_nonzero(np.linalg.norm(chunk, axis=1) > 0) for chunk in frame_vec_chunks)),
            "track_stats": {int(k): {k2: v2 for k2, v2 in v.items() if k2 != "boxes"} for k, v in track_stats.items()},
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
                    "representative_frame_path": x["frame_path"],
                    "objects": sorted(x["meta"]["objects"].keys()),
                    "appearances": x["meta"]["appearances"],
                    "tracks": x["meta"]["tracks"],
                    "detections": x["meta"]["detections"],
                    "deduplication": x["meta"].get("deduplication", ""),
                    "object_delta": x["meta"].get("object_delta", 0),
                }
                for x in frames
            ],
            "segments": segs,
        }
        # Semantic enrichment is a required stage. It is intentionally invoked
        # after evidence-frame writes complete, so every caption remains tied to
        # a durable source image and an exact segment interval.
        analyzer = NvidiaSemanticAnalyzer(
            api_key=os.getenv("NVIDIA_API_KEY", ""),
            base_url=os.getenv("NVIDIA_API_BASE_URL", ""),
            model=os.getenv("NVIDIA_VLM_MODEL", ""),
            timeout=float(os.getenv("NVIDIA_API_TIMEOUT", "30")),
        )
        semantic_started = time.perf_counter()
        semantic_total = len(self.idx["segments"])
        for semantic_index, segment in enumerate(self.idx["segments"], 1):
            semantic = analyzer.analyze(
                segment["frame_path"],
                objects=segment["objects"],
                tracks=segment["tracks"],
                start=segment["start"],
                end=segment["end"],
            ).to_dict()
            semantic["evidence_state"] = "semantic_description"
            semantic["claim_provenance"] = "nvidia_semantic"
            semantic["supporting_frame_path"] = segment["frame_path"]
            semantic["detector_context"] = {"objects": list(segment["objects"]), "tracks": list(segment["tracks"])}
            segment["semantic"] = semantic
            segment["tags"] = sorted(set(semantic["scene_tags"]) | set(semantic["event_tags"]))
            yield {"kind": "semantic_progress", "processed": semantic_index, "total": semantic_total,
                   "message": f"NVIDIA semantic analysis completed segment {semantic_index} of {semantic_total}."}
        timings["semantic_analysis"] = time.perf_counter() - semantic_started
        extractor = EventExtractor(self.settings.semantic_zones, self.settings.semantic_min_dwell_seconds)
        self.semantic_events = extractor.extract(self.idx["frames"], vr.width, vr.height)
        self.idx["events"] = self.semantic_events
        self.idx["meta"]["semantic_provider"] = "nvidia"
        self.idx["meta"]["semantic_segments"] = len(self.idx["segments"])
        self.idx["meta"]["event_count"] = len(self.semantic_events)
        from collections import Counter

        _obj_counter = Counter()
        for _row in self.idx["frames"]:
            for _obj in _row.get("objects", []):
                _obj_counter[_obj] += 1

        self.idx["meta"]["object_counts"] = dict(_obj_counter.most_common())
        self.idx["meta"]["total_detections"] = sum(len(row.get("detections", [])) for row in self.idx["frames"])
        self.idx["meta"]["unique_objects"] = len(_obj_counter)
        frame_chunks = list(zip(frame_vec_chunks, frame_id_chunks))
        seg_chunks = list(zip(seg_vec_chunks, seg_id_chunks))
        index_started = time.perf_counter()
        self.frame_idx.build_merged(frame_chunks, path=os.path.join(self.run_dir, "reports", "frame_index.tvim"))
        self.search_idx.build_merged(seg_chunks, path=os.path.join(self.run_dir, "reports", "segment_index.tvim"))
        # Build the crop embedding index only when explicitly enabled.
        if crop_vec_chunks:
            crop_chunks = list(zip(crop_vec_chunks, crop_id_chunks))
            self.crop_idx.build_merged(crop_chunks, path=os.path.join(self.run_dir, "reports", "crop_index.tvim"))
            self.crop_meta = crop_meta_list
        timings["index_building"] += time.perf_counter() - index_started
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
                        "tracks": x["tracks"],
                        "appearances": x["appearances"],
                        "detections": x["detections"],
                        "deduplication": x.get("deduplication", ""),
                        "object_delta": x.get("object_delta", 0),
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
                        "tracks": x["tracks"],
                        "temporal_stats": x["temporal_stats"],
                        "tags": x["tags"],
                        "semantic": x.get("semantic", {}),
                    }
                    for x in segs
                ],
                "events": self.semantic_events,
            },
        )
        # Generate zero-query analysis
        self._generate_zero_query(track_stats, fps, dur)
        total_elapsed = time.perf_counter() - t0
        timings["total"] = total_elapsed
        self.idx["meta"]["scan_timings"] = {k: round(v, 3) for k, v in timings.items()}
        logger.info(
            "scan_complete detection_s=%.1f frame_embedding_s=%.1f crop_embedding_s=%.1f "
            "frame_read_filter_s=%.1f indexed=%s exact_duplicate_skip=%s index_s=%.1f total_s=%.1f",
            timings["detection_tracking"], timings["frame_embeddings"], timings["crop_embeddings"],
            timings["frame_read_filter"], kept_frames, skipped_duplicates,
            timings["index_building"], total_elapsed,
        )
        yield {
            "kind": "done",
            "meta": {
                **self.idx["meta"],
                "retriever": self.frame_idx.backend,
                "segment_retriever": self.search_idx.backend,
                "verifier": self.ver.model_name,
                "scan_timings": {k: round(v, 3) for k, v in timings.items()},
            },
            "index_json": path,
        }

    def _generate_zero_query(self, track_stats, fps, dur):
        """Generate zero-query analysis: object inventory, event timeline, and summary."""
        # Object inventory: unique tracks per class with counts and dwell stats
        class_tracks = {}
        for tid, stats in track_stats.items():
            cname = stats["class_name"]
            if cname not in class_tracks:
                class_tracks[cname] = []
            class_tracks[cname].append({
                "track_id": tid,
                "dwell_time": stats["dwell_time"],
                "trajectory_length": stats["trajectory_length"],
                "entry_ts": stats["entry_ts"],
                "exit_ts": stats["exit_ts"],
                "avg_confidence": stats["avg_confidence"],
            })
        object_inventory = {
            cname: {
                "count": len(tracks),
                "total_dwell_time": round(sum(t["dwell_time"] for t in tracks), 2),
                "avg_dwell_time": round(sum(t["dwell_time"] for t in tracks) / max(1, len(tracks)), 2),
                "tracks": tracks,
            }
            for cname, tracks in sorted(class_tracks.items())
        }
        # Event timeline: high-motion windows + anomalies
        events = []
        for seg in self.idx.get("segments", []):
            ts_stats = seg.get("temporal_stats", {})
            avg_motion = ts_stats.get("avg_motion", 0)
            max_motion = ts_stats.get("max_motion", 0)
            if max_motion > 0.08:
                events.append({
                    "type": "high_motion",
                    "start": seg["start"],
                    "end": seg["end"],
                    "max_motion": max_motion,
                    "objects": seg["objects"],
                    "tracks": seg["tracks"],
                })
        # Anomalies: long dwell, sudden appearance
        for tid, stats in track_stats.items():
            if stats["dwell_time"] > dur * 0.5 and stats["trajectory_length"] > 5:
                events.append({
                    "type": "long_dwell",
                    "track_id": tid,
                    "class": stats["class_name"],
                    "dwell_time": stats["dwell_time"],
                    "entry_ts": stats["entry_ts"],
                    "exit_ts": stats["exit_ts"],
                })
            if stats["entry_ts"] > dur * 0.3 and stats["trajectory_length"] <= 2:
                events.append({
                    "type": "sudden_appearance",
                    "track_id": tid,
                    "class": stats["class_name"],
                    "entry_ts": stats["entry_ts"],
                })
        events.sort(key=lambda x: x.get("start", x.get("entry_ts", 0)))
        # Summary
        total_tracks = len(track_stats)
        classes = sorted(set(s["class_name"] for s in track_stats.values())) if track_stats else []
        summary_parts = [f"{dur:.1f}s video with {total_tracks} tracked objects"]
        if classes:
            summary_parts.append(f"detected classes: {', '.join(classes)}")
        if events:
            summary_parts.append(f"{len(events)} notable events detected")
        summary = " | ".join(summary_parts)
        zero_query = {
            "object_inventory": object_inventory,
            "event_timeline": events,
            "summary": summary,
            "meta": {
                "total_tracks": total_tracks,
                "classes": classes,
                "duration": dur,
                "fps": fps,
            },
        }
        self.zero_query = zero_query
        # Write to disk
        path = os.path.join(self.run_dir, "reports", "zero_query.json")
        self.rep.write_json(path, zero_query)
        return zero_query

    def warmup_models(self):
        self._warmup_failures = {}
        for name, fn in [("tracker", self.trk.load),
                         ("encoder", self.enc.load),
                         ("verifier", self.ver.warmup)]:
            try:
                fn()
            except Exception as e:
                self._warmup_failures[name] = str(e)
        self._warmup_done = True

    def warmup_status(self) -> str:
        if not self._warmup_done:
            if os.getenv("VISION_GUARD_SKIP_WARMUP") == "1":
                return "Video detector and retrieval models load on demand."
            return "Video models loading..."
        if not self._warmup_failures:
            return "Video detector and retrieval models ready."
        return "WARNING: " + " | ".join(
            f"{k} failed: {v}" for k, v in self._warmup_failures.items()
        )

    def verification_mode(self):
        if hasattr(self.ver, "verification_mode"):
            try:
                return self.ver.verification_mode()
            except Exception:
                return "unknown"
        backend = getattr(self.ver, "backend", "none")
        if backend == "nvidia_api":
            return "nvidia_api"
        if backend in {"unconfigured", "unavailable"}:
            return "nvidia_api_unconfigured" if backend == "unconfigured" else "nvidia_api_unavailable"
        return "unknown"

    def model_health(self, refresh=False):
        checked_at, cached = self._model_health_cache
        if not refresh and cached is not None and time.monotonic() - checked_at < 5.0:
            return cached
        snapshot = model_health_snapshot(self.model_provider)
        snapshot["detector"] = self.trk.model_status()
        try:
            analyzer = NvidiaSemanticAnalyzer(
                api_key=os.getenv("NVIDIA_API_KEY", ""), base_url=os.getenv("NVIDIA_API_BASE_URL", ""),
                model=os.getenv("NVIDIA_VLM_MODEL", ""), timeout=float(os.getenv("NVIDIA_API_TIMEOUT", "30")),
            )
            snapshot["semantic"] = analyzer.health()
        except SemanticAnalysisError as exc:
            snapshot["semantic"] = {"ready": False, "provider": "nvidia", "message": str(exc)}
        self._model_health_cache = (time.monotonic(), snapshot)
        return snapshot

    def _model_assisted_query(self, query, detector_labels):
        health = self.model_health().get("text_model", {})
        if not health.get("reachable"):
            return query, {
                "used": False,
                "provider": self.model_provider.provider_name,
                "message": health.get("message", "Selected text model is unavailable."),
            }
        label_list = sorted({str(label).strip().casefold() for label in detector_labels if str(label).strip()})
        prompt = (
            "Convert the CCTV search request into conservative JSON. "
            "detector_entities may contain only exact values from detector_labels. "
            "Do not invent observations, identities, frame numbers, or timestamps. "
            'Return keys detector_entities, events, attributes. '
            f"detector_labels={json.dumps(label_list)} request={json.dumps(query)}"
        )
        try:
            raw = self.model_provider.chat(
                [{"role": "system", "content": "You normalize search intent; you never claim evidence."},
                 {"role": "user", "content": prompt}],
                json_mode=True,
                temperature=0.1,
            )
            parsed = json.loads(raw)
        except (ModelProviderError, json.JSONDecodeError, TypeError) as exc:
            return query, {
                "used": False,
                "provider": self.model_provider.provider_name,
                "message": f"Model intent parsing failed; deterministic parsing used: {exc}",
            }
        allowed = set(label_list)
        entities = [
            str(value).strip().casefold() for value in parsed.get("detector_entities", [])
            if str(value).strip().casefold() in allowed
        ]
        safe_terms = []
        for key in ("events", "attributes"):
            values = parsed.get(key, [])
            if not isinstance(values, list):
                continue
            for value in values[:5]:
                term = " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))
                if term and len(term) <= 40:
                    safe_terms.append(term)
        assisted = " ".join([query, *entities, *safe_terms]).strip()
        return assisted, {
            "used": True,
            "provider": self.model_provider.provider_name,
            "message": "Text model normalized intent; retrieval still requires stored evidence.",
        }

    def embedding_mode(self):
        return self.enc.mode()

    def plan_query(self, query):
        try:
            detector_labels = tuple(self.trk.names().values())
        except Exception:
            detector_labels = ()
        assisted_query, model_assistance = self._model_assisted_query(query, detector_labels)
        plan = self.query_planner.plan(assisted_query, detector_labels=detector_labels)
        plan.query = query
        try:
            self.enc.load()
        except Exception:
            pass
        mode = self.embedding_mode()
        routes = set(plan.retrieval_routes)
        executable = True
        message = ""
        limitations = list(plan.limitations)

        if plan.clarification:
            executable = False
            message = plan.clarification
        elif "speech" in routes:
            executable = False
            message = "Speech queries are not available because this project does not index audio transcripts."
        elif "visual_semantic" in routes and mode != "semantic_embeddings":
            if self.ver.is_ready():
                routes.add("exhaustive_visual_verification")
                plan.retrieval_routes.append("exhaustive_visual_verification")
                limitations.append("Open query uses bounded verification across sampled indexed frames.")
            else:
                executable = False
                message = "This request needs semantic vision embeddings or a reachable vision verifier. Exact object queries can still use local detector evidence."
        elif not routes:
            executable = False
            message = "I could not map this request to an indexed object, attribute, action, or event. Please describe what should be visible."

        if mode == "metadata_embeddings":
            limitations.append("Semantic vision model unavailable; retrieval is grounded in detector/tracker metadata.")
        if not model_assistance["used"]:
            limitations.append(model_assistance["message"])
        payload = plan.to_dict()
        payload.update({
            "embedding_mode": mode,
            "executable": executable,
            "message": message,
            "limitations": sorted(set(limitations)),
            "model_assistance": model_assistance,
            "retrieval_query": assisted_query,
        })
        self.last_query_plan = payload
        self.last_query_message = message
        return payload

    def _exhaustive_visual_candidates(self, query):
        frames = self.idx.get("frames", [])
        if not frames:
            return []
        limit = self.settings.max_exhaustive_verification_frames
        if len(frames) <= limit:
            selected = frames
        else:
            positions = np.linspace(0, len(frames) - 1, num=limit, dtype=int)
            selected = [frames[int(position)] for position in positions]
        candidates = []
        for row in selected:
            start, end = self._clip_bounds(row["ts"])
            candidates.append({
                "query": query,
                "score": 0.0,
                "base_score": 0.0,
                "retrieval_mode": "exhaustive_visual_verification",
                "cache_key": f"open:{row['frame_id']}",
                "start": start,
                "end": end,
                "peak_ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row.get("objects", []),
                "tracks": row.get("tracks", []),
                "appearances": row.get("appearances", []),
                "det_boxes": [],
                "tags": ["open-query sampled-frame verification"],
                "summary": "Awaiting visual verification for an open query.",
            })
        return candidates

    def _temporal_track_hits(self, raw_q, q, qv, top_k):
        """Find hits for temporal queries using track trajectory features."""
        nq = f" {self._normalize_query(raw_q)} "
        hits = []
        dur = self.idx["meta"]["duration"]
        qobjs = set(self._q_objs(q))

        for tid, stats in self.track_stats.items():
            score = 0.0
            reason = ""

            # Loitering: long dwell time
            if " loitering " in nq or " loiter " in nq or " suspicious " in nq:
                if stats["dwell_time"] > max(dur * 0.2, 5.0):
                    score = 0.4 + min(0.3, stats["dwell_time"] / dur * 0.5)
                    reason = f"loitering ({stats['dwell_time']:.1f}s dwell)"

            # Enter/entry
            elif " enter " in nq or " entering " in nq or " entry " in nq:
                if stats["entry_ts"] > 0.5:
                    score = 0.45 + 0.1 * stats["avg_confidence"]
                    reason = f"entered at {stats['entry_ts']:.1f}s"

            # Exit/leave
            elif " exit " in nq or " exiting " in nq or " leave " in nq or " leaving " in nq:
                if stats["exit_ts"] < dur - 0.5:
                    score = 0.45 + 0.1 * stats["avg_confidence"]
                    reason = f"exited at {stats['exit_ts']:.1f}s"

            # Approach
            elif " approach " in nq or " approaching " in nq:
                if stats["trajectory_length"] >= 3:
                    score = 0.35 + 0.15 * stats["avg_confidence"]
                    reason = f"approaching ({stats['trajectory_length']} frames)"

            # Gather/crowd
            elif " gather " in nq or " gathering " in nq or " crowd " in nq or " crowded " in nq:
                # Check if multiple tracks overlap temporally
                overlaps = sum(
                    1 for other_tid, other in self.track_stats.items()
                    if other_tid != tid
                    and other["entry_ts"] <= stats["exit_ts"]
                    and other["exit_ts"] >= stats["entry_ts"]
                    and other["class_name"] == stats["class_name"]
                )
                if overlaps >= 2:
                    score = 0.35 + 0.08 * min(overlaps, 5)
                    reason = f"gathering ({overlaps + 1} {stats['class_name']} tracks overlapping)"

            if score < 0.3:
                continue

            # Filter by object class if specified
            if qobjs and stats["class_name"] not in qobjs:
                continue

            peak_ts = (stats["entry_ts"] + stats["exit_ts"]) / 2
            start, end = self._clip_bounds(peak_ts, pad=max(2.0, stats["dwell_time"] / 2))

            # Find best frame for this track
            frame_path = None
            for frame_row in self.idx.get("frames", []):
                if tid in frame_row.get("tracks", []):
                    frame_path = frame_row["frame_path"]
                    peak_ts = frame_row["ts"]
                    break

            if not frame_path:
                continue

            hits.append({
                "query": q,
                "score": round(score, 4),
                "base_score": round(score, 4),
                "retrieval_mode": "temporal_track",
                "cache_key": f"track:{tid}",
                "start": start,
                "end": end,
                "peak_ts": peak_ts,
                "representative_frame_path": frame_path,
                "frame_path": frame_path,
                "objects": [stats["class_name"]],
                "tracks": [tid],
                "det_boxes": [],
                "tags": [reason],
                "summary": f"track-based temporal match: {reason} | {stats['class_name']} (track {tid})",
                "low_confidence": True,
            })

        hits = sorted(hits, key=lambda x: x["score"], reverse=True)
        # Deduplicate by time
        deduped = []
        for hit in hits:
            if len(deduped) >= top_k:
                break
            if any(abs(hit["peak_ts"] - x["peak_ts"]) < 2.0 for x in deduped):
                continue
            deduped.append(hit)
        return deduped

    def _candidate_hits(self, raw_q, top_k=4):
        query_plan = self.plan_query(raw_q)
        retrieval_query = query_plan.get("retrieval_query") or raw_q
        if not query_plan["executable"]:
            dimension = max(1, int(getattr(self.enc, "fallback_dimension", 1)))
            return self._normalize_query(raw_q), np.zeros((dimension,), dtype=np.float32), query_plan.get("entities", []), [], 0
        if "exhaustive_visual_verification" in query_plan.get("retrieval_routes", []):
            candidates = self._exhaustive_visual_candidates(self._normalize_query(raw_q))
            dimension = max(1, int(getattr(self.enc, "fallback_dimension", 1)))
            return self._normalize_query(raw_q), np.zeros((dimension,), dtype=np.float32), [], candidates, len(candidates)
        q = self._normalize_query(retrieval_query)
        qv = self._embed_query(retrieval_query)
        qobjs = self._q_objs(q)
        qcolors = set(self._query_colors(q))
        # Handle temporal queries via track trajectory features
        if self._is_temporal_query(raw_q) and self.track_stats:
            hits = self._temporal_track_hits(raw_q, q, qv, top_k)
            if hits:
                return q, qv, qobjs, hits, min(4, len(hits))
        detector_hits = self._refine_detector_hits(q, top_k)
        if detector_hits:
            hits = self._apply_reselection(detector_hits, q, qv, top_n=min(4, len(detector_hits)))
            return q, qv, qobjs, hits, min(2, len(hits))
        frames = self.idx.get("frames", [])
        frame_map = {int(x["frame_id"]): x for x in frames}
        fetch_k = min(max(top_k * 12, 36), len(frames))
        frame_scores, frame_ids = self.frame_idx.search(qv, fetch_k)
        frame_retrieval_mode = "metadata_frame" if self.embedding_mode() == "metadata_embeddings" else "semantic_frame"
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
            rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "retrieval_mode": frame_retrieval_mode,
                "frame_id": row["frame_id"],
                "ts": row["ts"],
                "representative_frame_path": row["frame_path"],
                "frame_path": row["frame_path"],
                "objects": row["objects"],
                "appearances": row.get("appearances", []),
                "tracks": row["tracks"],
                "detections": row.get("detections", []),
            })
        ranked_rows = sorted(rows, key=lambda x: x["score"], reverse=True)
        rows = [x for x in ranked_rows if x["score"] >= 0.14]
        out = self._cluster_frame_hits(rows, top_k=top_k, gap_sec=max(self.idx["meta"]["frame_interval_sec"] * 1.25, 1.0))
        if out:
            out = self._apply_reselection(out, q, qv, top_n=min(4, len(out)))
            verify_n = min(8, len(out)) if not qobjs else min(4, len(out))
            return q, qv, qobjs, out, verify_n
        obj_hits = self._fallback_object_hits(q, top_k)
        if obj_hits:
            obj_hits = self._apply_reselection(obj_hits, q, qv, top_n=min(4, len(obj_hits)))
            return q, qv, qobjs, obj_hits, min(4, len(obj_hits))
        if ranked_rows and not self._is_strict_object_query(q):
            weak = self._cluster_frame_hits(ranked_rows[: max(top_k * 3, 8)], top_k=top_k, gap_sec=max(self.idx["meta"]["frame_interval_sec"] * 1.25, 1.0))
            for hit in weak:
                hit["summary"] = f"low-confidence visual match at {hit['peak_ts']:.2f}s | detected: {', '.join(hit['objects']) if hit['objects'] else 'no tracked objects'}"
                hit["low_confidence"] = True
                hit["retrieval_mode"] = "weak_semantic"
            if weak:
                weak = self._apply_reselection(weak, q, qv, top_n=min(4, len(weak)))
                verify_n = min(8, len(weak)) if not qobjs else 1
                return q, qv, qobjs, weak, verify_n
        n = len(self.idx["segments"])
        if n == 0:
            return q, qv, qobjs, [], 0
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
            seg_rows.append({
                "query": q,
                "score": score,
                "base_score": float(base_score),
                "retrieval_mode": "metadata_segment" if self.embedding_mode() == "metadata_embeddings" else "semantic_segment",
                "cache_key": f"seg:{int(seg['seg_id'])}",
                "start": seg["start"],
                "end": seg["end"],
                "peak_ts": seg["mid"],
                "representative_frame_path": seg["frame_path"],
                "frame_path": seg["frame_path"],
                "objects": seg["objects"],
                "tracks": seg["tracks"],
                "det_boxes": [],
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
        if out:
            out = self._apply_reselection(out, q, qv, top_n=min(4, len(out)))
            verify_n = min(8, len(out)) if not qobjs else min(4, len(out))
            return q, qv, qobjs, out, verify_n
        return q, qv, qobjs, [], 0

    def _wait_for_verifier(self):
        if self.ver.backend not in (None, "none") or self.verifier_ready_timeout <= 0:
            return
        deadline = time.monotonic() + self.verifier_ready_timeout
        while self.ver.backend in (None, "none"):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(self.verifier_poll_interval, remaining))

    def search_stream(self, raw_q, top_k=4):
        """Compatibility iterator backed by the same LangGraph query path."""
        yield self.search(raw_q, top_k=top_k)

    def _graph_object_evidence(self, plan, top_k):
        return self._refine_detector_hits(plan["normalized_query"], top_k)

    def _graph_count_evidence(self, plan, top_k):
        wanted = set(plan.get("entities", []))
        if not wanted:
            return []
        by_track = {}
        for frame in self.idx.get("frames", []):
            for detection in frame.get("detections", []):
                track_id = detection.get("track_id")
                if track_id is None or detection.get("name") not in wanted:
                    continue
                candidate = (float(detection.get("conf", 0.0)), frame, detection)
                if track_id not in by_track or candidate[0] > by_track[track_id][0]:
                    by_track[int(track_id)] = candidate
        count = len(by_track)
        if not count:
            return []
        rows = []
        for track_id, (confidence, frame, detection) in sorted(by_track.items()):
            rows.append({
                "query": plan["query"], "score": confidence, "retrieval_mode": "distinct_track_count",
                "cache_key": f"count:{track_id}:{frame['frame_id']}", "start": frame["ts"], "end": frame["ts"],
                "peak_ts": frame["ts"], "frame_path": frame["frame_path"],
                "representative_frame_path": frame["frame_path"], "objects": [detection["name"]],
                "tracks": [track_id], "det_boxes": [detection.get("box", [])], "count": count,
                "summary": f"Distinct tracked {detection['name']} {track_id}",
                "evidence_state": "detector_fact", "claim_provenance": "yolo_botsort",
            })
        return rows[:max(top_k, count)]

    def _graph_temporal_evidence(self, plan, top_k):
        query = plan["normalized_query"]
        relation = plan.get("temporal_relation", "none")
        numeric = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", query)]
        lower, upper = 0.0, float(self.idx.get("meta", {}).get("duration", 0.0))
        if relation == "between" and len(numeric) >= 2:
            lower, upper = sorted(numeric[:2])
        elif relation == "before" and numeric:
            upper = numeric[0]
        elif relation == "after" and numeric:
            lower = numeric[0]
        else:
            terms = set(query.split())
            if {"enter", "entered", "entering"} & terms:
                required_type = "zone_entered"
            elif {"exit", "exited", "exiting", "leave", "leaving"} & terms:
                required_type = "zone_exited"
            elif {"dwell", "stay", "stayed"} & terms:
                required_type = "track_dwell"
            elif {"move", "moved", "movement"} & terms:
                required_type = "track_moved"
            else:
                return []
            reference_clause = query.split(relation, 1)[1].strip() if relation in query else ""
            detector_labels = sorted({label for frame in self.idx.get("frames", []) for label in frame.get("objects", [])})
            reference_entities = set(self.query_planner.resolve_entities(reference_clause, detector_labels))
            reference_events = [
                event for event in self.idx.get("events", [])
                if event.get("type") == required_type
                and (not reference_entities or event.get("class_name") in reference_entities)
            ]
            if not reference_events:
                return []
            timestamps = sorted(float(event["timestamp"]) for event in reference_events)
            if relation == "before":
                upper = timestamps[0]
            elif relation == "after":
                lower = timestamps[-1]
            else:
                return []
        target_clause = query.split(relation, 1)[0].strip()
        target_plan = self.query_planner.plan(target_clause, sorted({label for frame in self.idx.get("frames", []) for label in frame.get("objects", [])}))
        candidates = self._graph_object_evidence(target_plan.to_dict(), max(top_k * 8, 32)) if target_plan.entities else []
        return [row for row in candidates if lower <= float(row["peak_ts"]) <= upper][:top_k]

    @staticmethod
    def _group_frames_by_time(frames, window_seconds):
        groups = []
        for frame in sorted(frames, key=lambda row: float(row["ts"])):
            if not groups or float(frame["ts"]) - float(groups[-1][0]["ts"]) >= float(window_seconds):
                groups.append([frame])
            else:
                groups[-1].append(frame)
        return groups

    def _graph_event_evidence(self, plan, top_k):
        query_terms = set(plan["normalized_query"].split())
        wanted_objects = set(plan.get("entities", []))
        required_types = set()
        if {"enter", "entered", "entering"} & query_terms:
            required_types.add("zone_entered")
        if {"exit", "exited", "exiting"} & query_terms:
            required_types.add("zone_exited")
        if {"dwell", "stay", "stayed"} & query_terms:
            required_types.add("track_dwell")
        if {"move", "moved", "movement", "direction"} & query_terms:
            required_types.add("track_moved")
        events = []
        for event in self.idx.get("events", []):
            event_terms = set(event["type"].replace("_", " ").split())
            if wanted_objects and event.get("class_name") not in wanted_objects:
                continue
            if required_types and event["type"] not in required_types:
                continue
            if query_terms and not (query_terms & event_terms or {"enter", "exit", "movement", "dwell", "stay"} & query_terms):
                continue
            events.append({"query": plan["query"], "score": 1.0, "retrieval_mode": "event_graph",
                           "cache_key": f"event:{event['event_id']}", "start": event["timestamp"], "end": event["timestamp"],
                           "peak_ts": event["timestamp"], "frame_path": event["frame_path"],
                           "representative_frame_path": event["frame_path"], "objects": [event["class_name"]],
                           "tracks": [event["track_id"]], "tags": [event["type"]], "event": event,
                           "evidence_state": "event_fact", "claim_provenance": "yolo_botsort_event_graph",
                           "summary": f"{event['type']} for track {event['track_id']} at {event['timestamp']:.3f}s"})
        return events[:top_k]

    def _graph_zone_evidence(self, plan, top_k):
        terms = set(plan["normalized_query"].split())
        return [row for row in self._graph_event_evidence(plan, max(top_k * 4, 12))
                if row["event"]["type"] in {"zone_entered", "zone_exited"}
                and (not row["event"].get("zone") or row["event"]["zone"].casefold() in terms)][:top_k]

    def _graph_semantic_evidence(self, plan, top_k):
        if not self.idx.get("segments"):
            return []
        query_vector = self._embed_query(plan["normalized_query"])
        scores, segment_ids = self.search_idx.search(query_vector, k=max(1, top_k))
        by_id = {int(segment["seg_id"]): segment for segment in self.idx.get("segments", [])}
        rows = []
        for score, segment_id in zip(scores, segment_ids):
            if float(score) < self.settings.minimum_vector_similarity:
                continue
            segment = by_id.get(int(segment_id))
            if segment is None:
                continue
            semantic = segment.get("semantic") or {}
            rows.append({"query": plan["query"], "score": float(score), "retrieval_mode": "vector_segment",
                         "segment_id": int(segment["seg_id"]),
                         "cache_key": f"semantic:{int(segment['seg_id'])}", "start": segment["start"], "end": segment["end"],
                         "peak_ts": segment["mid"], "frame_path": segment["frame_path"],
                         "representative_frame_path": segment["frame_path"], "objects": segment["objects"], "tracks": segment["tracks"],
                         "tags": segment.get("tags", []), "semantic": semantic, "summary": semantic.get("caption", ""),
                         "evidence_state": "semantic_description", "claim_provenance": "nvidia_semantic"})
        return sorted(rows, key=lambda row: row["score"], reverse=True)[:top_k]

    def _graph_verify_evidence(self, query, evidence):
        if not evidence:
            return {"state": "no_candidates", "confirmed": False, "evidence": []}
        if not self.ver.is_ready():
            return {"state": "unavailable", "confirmed": False, "evidence": []}
        confirmed = []
        video_identity = os.path.abspath(self.idx.get("video", "")) if self.idx else ""
        for row in evidence:
            frame_path = row.get("representative_frame_path") or row.get("frame_path")
            result = self.ver.verify_query(
                frame_path,
                query,
                frame_key=f"{video_identity}:{row.get('cache_key', frame_path)}",
            )
            if not result.get("matched"):
                continue
            verified = dict(row)
            verified["verified_match"] = True
            verified["verification_mode"] = result.get("verification_mode", self.verification_mode())
            verified["verify_score"] = float(result.get("confidence", 0.0))
            verified["verified_caption"] = str(result.get("caption", ""))
            verified["det_boxes"] = list(result.get("boxes", []))
            verified["evidence_state"] = "verified_claim"
            verified["claim_provenance"] = verified["verification_mode"]
            confirmed.append(verified)
        return {
            "state": "confirmed" if confirmed else "rejected",
            "confirmed": bool(confirmed),
            "evidence": confirmed,
        }

    def capture_snapshot(self):
        """Capture all query-critical state without sharing mutable vector indexes."""
        if not self.idx:
            raise RuntimeError("Cannot snapshot an empty index.")
        return {
            "idx": self.idx,
            "frame_idx": self.frame_idx.snapshot(),
            "search_idx": self.search_idx.snapshot(),
            "crop_idx": self.crop_idx.snapshot(),
            "crop_meta": list(self.crop_meta),
            "semantic_events": list(self.semantic_events),
            "track_stats": dict(self.track_stats),
            "run_dir": self.run_dir,
            "zero_query": self.zero_query,
            "clip": self.clip,
            "rep": self.rep,
        }

    def activate_snapshot(self, snapshot):
        """Atomically restore one video's immutable query/export state."""
        self.idx = snapshot["idx"]
        self.frame_idx = snapshot["frame_idx"]
        self.search_idx = snapshot["search_idx"]
        self.crop_idx = snapshot["crop_idx"]
        self.crop_meta = list(snapshot["crop_meta"])
        self.semantic_events = list(snapshot["semantic_events"])
        self.track_stats = dict(snapshot["track_stats"])
        self.run_dir = snapshot["run_dir"]
        self.zero_query = snapshot["zero_query"]
        self.clip = snapshot["clip"]
        self.rep = snapshot["rep"]

    def search(self, q, top_k=4):
        if not self.idx:
            return []
        detector_labels = sorted({label for frame in self.idx.get("frames", []) for label in frame.get("objects", [])})
        result = self.query_graph.invoke(q.strip(), detector_labels, top_k=top_k)
        self.last_query_plan = result.get("plan")
        self.last_query_message = result.get("answer", "")
        return result.get("evidence", [])

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
            row["representative_frame_path"] = row.get("representative_frame_path") or row.get("frame_path")
            row["gallery_frame"] = row["representative_frame_path"]
            if i == 1:
                row = self._attach_gallery_frame(row, query)
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
        seg_clip, frames, seen = self.seg.segment_clip(raw, query, seg_mp4, seg_dir, stride=3, fallback_boxes=row.get("det_boxes", []))
        return {"raw_clip": raw, "clip": seg_clip if seen > 0 else raw, "frames": frames, "seen": seen}

    def _start_segment(self, row, query):
        if row["segmented"] or row["match_id"] in self.seg_jobs:
            return
        self.seg_jobs[row["match_id"]] = self.pool.submit(self._segment_payload, dict(row), query)

    def _raw_fallback_payload(self, row, reason):
        raw = self._ensure_raw_clip(row, wait=True)
        row["raw_clip"] = raw
        row["clip"] = raw
        row["frames"] = []
        row["segmented"] = False
        row["export_mode"] = "raw_fallback"
        row["export_warning"] = f"Raw clip export fallback; segmentation {reason}."
        if row["export_warning"] not in row["summary"]:
            row["summary"] = f"{row['summary']} | {row['export_warning']}"
        return row

    def _ensure_segment(self, row, query, timeout=None, allow_raw_fallback=False):
        if row["segmented"]:
            return row
        job = self.seg_jobs.get(row["match_id"])
        if job is None:
            if timeout is None:
                payload = self._segment_payload(row, query)
            else:
                job = self.pool.submit(self._segment_payload, dict(row), query)
                self.seg_jobs[row["match_id"]] = job
                try:
                    payload = job.result(timeout=timeout)
                except TimeoutError:
                    if allow_raw_fallback:
                        return self._raw_fallback_payload(row, "timed out")
                    raise
                except Exception as exc:
                    if allow_raw_fallback:
                        return self._raw_fallback_payload(row, f"failed: {exc}")
                    raise
        else:
            try:
                payload = job.result(timeout=timeout)
            except TimeoutError:
                if allow_raw_fallback:
                    return self._raw_fallback_payload(row, "timed out")
                raise
            except Exception as exc:
                if allow_raw_fallback:
                    return self._raw_fallback_payload(row, f"failed: {exc}")
                raise
        row["raw_clip"] = payload["raw_clip"]
        row["clip"] = payload["clip"]
        row["frames"] = payload["frames"]
        row["segmented"] = bool(payload["seen"] > 0)
        row["export_mode"] = "segmented" if row["segmented"] else "raw_fallback"
        if payload["seen"] == 0 and "no grounded mask, showing raw clip" not in row["summary"]:
            row["summary"] = f"{row['summary']} | no grounded mask, showing raw clip"
        return row

    def export_selected_detailed(self, picks, query, segment_timeout=20):
        rows = [x for x in self.last_hits if x["label"] in picks]
        if not rows:
            return {
                "ok": False,
                "message": "No selected matches to export.",
                "files": {},
                "rows": [],
                "export_mode": "none",
                "warnings": [],
            }
        for row in rows:
            self._ensure_segment(row, query, timeout=segment_timeout, allow_raw_fallback=True)
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        js = self.rep.write_json(os.path.join(self.run_dir, "reports", f"selected_{base}.json"), {"hits": rows})
        csv = self.rep.write_csv(os.path.join(self.run_dir, "reports", f"selected_{base}.csv"), rows)
        html = self.rep.write_html(os.path.join(self.run_dir, "reports", f"selected_{base}.html"), {"query": rows[0]["query"], "video": self.idx["video"], "hits": rows})
        zipf = self.rep.write_zip(os.path.join(self.run_dir, "reports", f"selected_{base}.zip"), [x["clip"] for x in rows] + [x["raw_clip"] for x in rows])
        warnings = [x.get("export_warning") for x in rows if x.get("export_warning")]
        mode = "segmented" if all(x.get("segmented") for x in rows) else "raw_fallback"
        return {
            "ok": True,
            "message": "Export created." if mode == "segmented" else "Raw clip export fallback; segmentation unavailable or timed out.",
            "files": {"zip": zipf, "html": html, "csv": csv, "json": js},
            "rows": rows,
            "export_mode": mode,
            "warnings": warnings,
        }

    def export_selected(self, picks, query):
        result = self.export_selected_detailed(picks, query)
        if not result["ok"]:
            return None, None, None
        zipf = result["files"].get("zip")
        html = result["files"].get("html")
        csv = result["files"].get("csv")
        return zipf, html, csv
