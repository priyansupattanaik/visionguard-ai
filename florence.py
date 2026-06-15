import os

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class FlorenceVerifier:
    def __init__(self, model="microsoft/Florence-2-base", device=None, task="<MORE_DETAILED_CAPTION>"):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.p = None
        self.m = None
        self.cache = {}
        self.failed = False

    def load(self):
        if self.m is not None or self.failed:
            return
        try:
            dtype = torch.float16 if self.dev == "cuda" else torch.float32
            self.p = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.m = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map="auto" if self.dev == "cuda" else None,
            )
            if self.dev != "cuda":
                self.m.to(self.dev)
            self.m.eval()
        except Exception:
            self.failed = True
            self.p = None
            self.m = None

    def _decode(self, generated, size):
        text = self.p.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.p.post_process_generation(text, task=self.task, image_size=size)
        if isinstance(parsed, dict):
            val = parsed.get(self.task)
            if isinstance(val, str):
                return val.strip()
        return text.replace(self.task, "").strip()

    def caption_path(self, frame_path):
        if frame_path in self.cache:
            return self.cache[frame_path]
        self.load()
        if self.m is None or self.p is None:
            return ""
        if not os.path.exists(frame_path):
            return ""
        frame = cv2.imread(frame_path)
        if frame is None:
            return ""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = img.size
        inp = self.p(text=self.task, images=img, return_tensors="pt")
        inp = {k: v.to(self.m.device) if hasattr(v, "to") else v for k, v in inp.items()}
        try:
            with torch.no_grad():
                out = self.m.generate(**inp, max_new_tokens=96, num_beams=3, do_sample=False)
            caption = self._decode(out, (w, h))
        except Exception:
            caption = ""
        self.cache[frame_path] = caption
        return caption
