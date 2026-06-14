import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor


class GroundedSegmenter:
    def __init__(self, gdino="IDEA-Research/grounding-dino-base", sam="facebook/sam2.1-hiera-small", device=None):
        self.gdino_name = gdino
        self.sam_name = sam
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dp = None
        self.dm = None
        self.sp = None
        self.sm = None

    def load(self):
        if self.dm is None:
            self.dp = AutoProcessor.from_pretrained(self.gdino_name)
            self.dm = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_name, device_map="auto" if self.dev == "cuda" else None)
            if self.dev != "cuda":
                self.dm.to(self.dev)
        if self.sm is None:
            self.sp = Sam2Processor.from_pretrained(self.sam_name)
            self.sm = Sam2Model.from_pretrained(self.sam_name, device_map="auto" if self.dev == "cuda" else None)
            if self.dev != "cuda":
                self.sm.to(self.dev)

    def detect(self, frame, query, box_thr=0.28, text_thr=0.2):
        self.load()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        labels = [[query.strip().lower()]]
        inp = self.dp(images=img, text=labels, return_tensors="pt").to(self.dm.device)
        with torch.no_grad():
            out = self.dm(**inp)
        res = self.dp.post_process_grounded_object_detection(
            out,
            inp.input_ids,
            threshold=box_thr,
            text_threshold=text_thr,
            target_sizes=[(frame.shape[0], frame.shape[1])],
        )[0]
        boxes = []
        scores = []
        texts = []
        for box, score, text in zip(res["boxes"], res["scores"], res.get("text_labels", res.get("labels", []))):
            boxes.append([float(x) for x in box.tolist()])
            scores.append(float(score))
            texts.append(str(text))
        return boxes, scores, texts

    def segment(self, frame, boxes):
        self.load()
        if not boxes:
            return []
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.sp(images=img, input_boxes=[boxes], return_tensors="pt").to(self.sm.device)
        with torch.no_grad():
            out = self.sm(**inp, multimask_output=False)
        masks = self.sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0]
        res = []
        for i in range(masks.shape[0]):
            mask = masks[i, 0].numpy() if masks.ndim == 4 else masks[i].numpy()
            res.append(mask > 0)
        return res

    def overlay(self, frame, boxes, scores, masks):
        out = frame.copy()
        cols = [(40, 220, 120), (220, 120, 40), (60, 140, 240), (220, 60, 170)]
        for i, box in enumerate(boxes):
            c = cols[i % len(cols)]
            x1, y1, x2, y2 = [int(v) for v in box]
            if i < len(masks):
                m = masks[i]
                lay = np.zeros_like(out)
                lay[m] = c
                out = cv2.addWeighted(out, 1.0, lay, 0.35, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            cv2.putText(out, f"{scores[i]:.2f}", (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        return out

    def segment_clip(self, video, query, out_path, frame_dir, stride=3):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        tmp = f"{out_path}.part.mp4"
        if os.path.exists(tmp):
            os.remove(tmp)
        out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        prev = None
        picks = []
        seen = 0
        first = None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first is None:
                first = frame.copy()
            if i % stride == 0:
                boxes, scores, _ = self.detect(frame, query)
                masks = self.segment(frame, boxes[:2]) if boxes else []
                prev = self.overlay(frame, boxes[:2], scores[:2], masks)
                if boxes:
                    seen += 1
                    p = os.path.join(frame_dir, f"seg_{i:05d}.jpg")
                    cv2.imwrite(p, prev)
                    picks.append(p)
            out.write(prev if prev is not None else frame)
            i += 1
        cap.release()
        out.release()
        os.replace(tmp, out_path)
        if seen == 0 and first is not None:
            p = os.path.join(frame_dir, "fallback_00000.jpg")
            cv2.imwrite(p, first)
            picks.append(p)
        return out_path, picks, seen
