import json
import os
import re

import torch
from PIL import Image


class QwenFrameVerifier:
    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.process_vision_info = None
        self.failed = False
        self.cache = {}

    def load(self):
        if self.model is not None or self.failed:
            return
        try:
            from qwen_vl_utils import process_vision_info
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            dtype = torch.float16 if self.dev == "cuda" else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self.dev == "cuda" else None,
                trust_remote_code=True,
            )
            if self.dev != "cuda":
                self.model.to(self.dev)
            self.model.eval()
            self.process_vision_info = process_vision_info
        except Exception:
            self.failed = True
            self.model = None
            self.processor = None
            self.process_vision_info = None

    def _extract_json(self, text):
        if not text:
            return {}
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    def _clean_boxes(self, boxes, size):
        w, h = size
        clean = []
        for box in boxes or []:
            if isinstance(box, dict):
                box = box.get("box") or box.get("bbox") or box.get("coordinates")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            vals = [float(x) for x in box]
            if max(vals) <= 1.5:
                vals = [vals[0] * w, vals[1] * h, vals[2] * w, vals[3] * h]
            if max(vals) <= 1000.0 and (vals[2] > w or vals[3] > h):
                vals = [vals[0] / 1000.0 * w, vals[1] / 1000.0 * h, vals[2] / 1000.0 * w, vals[3] / 1000.0 * h]
            x1, y1, x2, y2 = vals
            x1 = max(0.0, min(w, x1))
            x2 = max(0.0, min(w, x2))
            y1 = max(0.0, min(h, y1))
            y2 = max(0.0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            clean.append([x1, y1, x2, y2])
        return clean

    def _ask(self, frame_path, prompt, max_new_tokens=180):
        self.load()
        if self.model is None or self.processor is None or self.process_vision_info is None:
            return ""
        if not frame_path or not os.path.exists(frame_path):
            return ""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": frame_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = self.process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated = generated[:, inputs.input_ids.shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        except Exception:
            return ""

    def verify_query(self, frame_path, query):
        key = ("verify", frame_path, query.strip().lower())
        if key in self.cache:
            return self.cache[key]
        image = Image.open(frame_path).convert("RGB") if frame_path and os.path.exists(frame_path) else None
        if image is None:
            return {"matched": False, "confidence": 0.0, "caption": "", "boxes": []}
        prompt = (
            "You are verifying CCTV search results. "
            "Decide whether the image clearly satisfies the exact user query. "
            "For events, answer true only if the event is visible in this frame. "
            "Do not infer beyond the visible evidence. "
            f"User query: {query}\n"
            "Return JSON only with keys: matched(boolean), confidence(number 0 to 1), "
            "description(short string), boxes(list of [x1,y1,x2,y2] pixel boxes for visible matching regions, empty if not localizable)."
        )
        raw = self._ask(frame_path, prompt)
        data = self._extract_json(raw)
        boxes = self._clean_boxes(data.get("boxes", []), image.size)
        matched = bool(data.get("matched", False))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        if confidence < 0.45:
            matched = False
        result = {
            "matched": matched,
            "confidence": max(0.0, min(1.0, confidence)),
            "caption": str(data.get("description", "") or "").strip(),
            "boxes": boxes,
        }
        self.cache[key] = result
        return result

    def ground_phrase(self, frame_path, phrase, multi=True):
        result = self.verify_query(frame_path, phrase)
        return result.get("boxes", []) if result.get("matched") else []
