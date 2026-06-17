import os
import re

import torch
from PIL import Image


class LocateAnythingVerifier:
    def __init__(self, model="nvidia/LocateAnything-3B", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = None
        self.failed = False
        self.caption_cache = {}
        self.ground_cache = {}

    def load(self):
        if self.pipe is not None or self.failed:
            return
        try:
            from transformers import pipeline

            dtype = torch.float16 if self.dev == "cuda" else torch.float32
            kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "torch_dtype": dtype,
            }
            if self.dev == "cuda":
                kwargs["device_map"] = "auto"
            else:
                kwargs["device"] = -1
            self.pipe = pipeline("image-text-to-text", **kwargs)
        except Exception:
            self.failed = True
            self.pipe = None

    def _extract_text(self, output):
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict):
            for key in ("generated_text", "text", "answer", "content"):
                val = output.get(key)
                if isinstance(val, str):
                    return val.strip()
            return ""
        if isinstance(output, list) and output:
            return self._extract_text(output[0])
        return ""

    def _run_prompt(self, image, prompt, max_new_tokens=256):
        self.load()
        if self.pipe is None:
            return ""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            out = self.pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
        except TypeError:
            out = self.pipe(text=messages, max_new_tokens=max_new_tokens)
        except Exception:
            return ""
        return self._extract_text(out)

    def _open(self, frame_path):
        if not frame_path or not os.path.exists(frame_path):
            return None
        try:
            return Image.open(frame_path).convert("RGB")
        except Exception:
            return None

    def _parse_boxes(self, answer, size):
        if not answer:
            return []
        w, h = size
        boxes = []
        for match in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(x) for x in match.groups()]
            px = [
                max(0.0, min(w, x1 / 1000.0 * w)),
                max(0.0, min(h, y1 / 1000.0 * h)),
                max(0.0, min(w, x2 / 1000.0 * w)),
                max(0.0, min(h, y2 / 1000.0 * h)),
            ]
            if px[2] <= px[0] or px[3] <= px[1]:
                continue
            boxes.append(px)
        return boxes

    def describe_frame(self, frame_path):
        if frame_path in self.caption_cache:
            return self.caption_cache[frame_path]
        image = self._open(frame_path)
        if image is None:
            return ""
        text = self._run_prompt(image, "Describe this image in one short sentence.", max_new_tokens=80)
        self.caption_cache[frame_path] = text
        return text

    def ground_phrase(self, frame_path, phrase, multi=True):
        key = (frame_path, phrase.strip().lower(), bool(multi))
        if key in self.ground_cache:
            return self.ground_cache[key]
        image = self._open(frame_path)
        if image is None:
            return []
        phrase = phrase.strip().lower()
        if not phrase:
            return []
        if multi:
            prompt = f"Locate all the instances that match the following description: {phrase}."
        else:
            prompt = f"Locate a single instance that matches the following description: {phrase}."
        answer = self._run_prompt(image, prompt, max_new_tokens=160)
        boxes = self._parse_boxes(answer, image.size)
        self.ground_cache[key] = boxes
        return boxes

    def verify_query(self, frame_path, query):
        boxes = self.ground_phrase(frame_path, query, multi=True)
        caption = self.describe_frame(frame_path) if not boxes else ""
        return {
            "boxes": boxes,
            "caption": caption,
            "matched": bool(boxes),
        }
