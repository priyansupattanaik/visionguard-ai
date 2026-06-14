import re

import cv2
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer


class LocateAnythingGrounder:
    def __init__(self, model="nvidia/LocateAnything-3B", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.dev == "cuda" else torch.float32
        self.tok = None
        self.proc = None
        self.model = None
        self.failed = False

    def load(self):
        if self.model is not None or self.failed:
            return
        try:
            self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.proc = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.dev).eval()
        except Exception:
            self.failed = True
            self.tok = None
            self.proc = None
            self.model = None

    def _to_pil(self, frame):
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _prompt(self, query):
        return f"Locate all the instances that match the following description: {query.strip()}."

    def _parse_boxes(self, answer, w, h):
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(v) for v in m.groups()]
            boxes.append([
                x1 / 1000.0 * w,
                y1 / 1000.0 * h,
                x2 / 1000.0 * w,
                y2 / 1000.0 * h,
            ])
        return boxes

    def detect(self, frame, query, generation_mode="hybrid", max_new_tokens=1024):
        self.load()
        if self.model is None or self.proc is None or self.tok is None:
            return [], [], ""
        img = self._to_pil(frame)
        prompt = self._prompt(query)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.proc.process_vision_info(messages)
        inp = self.proc(text=[text], images=images, videos=videos, return_tensors="pt").to(self.dev)
        pixel_values = inp["pixel_values"].to(self.dtype)
        image_grid_hws = inp.get("image_grid_hws", None)
        with torch.no_grad():
            out = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inp["input_ids"],
                attention_mask=inp["attention_mask"],
                image_grid_hws=image_grid_hws,
                tokenizer=self.tok,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=0.0,
                do_sample=False,
                top_p=1.0,
                repetition_penalty=1.0,
                verbose=False,
            )
        answer = out[0] if isinstance(out, tuple) else out
        answer = answer if isinstance(answer, str) else str(answer)
        h, w = frame.shape[:2]
        boxes = self._parse_boxes(answer, w, h)
        scores = [1.0] * len(boxes)
        return boxes, scores, answer
