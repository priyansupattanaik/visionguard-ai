import cv2
import json

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, Qwen2_5_VLForConditionalGeneration


class SearchEncoder:
    def __init__(self, model="google/siglip2-base-patch16-224", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.p = None
        self.m = None

    def load(self):
        if self.m is not None:
            return
        self.p = AutoProcessor.from_pretrained(self.model_name)
        self.m = AutoModel.from_pretrained(self.model_name, device_map="auto" if self.dev == "cuda" else None)
        if self.dev != "cuda":
            self.m.to(self.dev)
        self.m.eval()

    def _vec(self, x):
        if hasattr(x, "pooler_output"):
            x = x.pooler_output
        elif hasattr(x, "image_embeds"):
            x = x.image_embeds
        elif hasattr(x, "text_embeds"):
            x = x.text_embeds
        return x

    def _norm(self, x):
        x = self._vec(x).detach().cpu().numpy()[0]
        n = np.linalg.norm(x)
        if n == 0:
            return x.astype(np.float32)
        return (x / n).astype(np.float32)

    def embed_text(self, txt):
        self.load()
        txt = f"this is a photo of {txt.strip().lower()}"
        inp = self.p(text=[txt], return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vec = self.m.get_text_features(**inp)
        return self._norm(vec)

    def embed_frame(self, frame):
        self.load()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inp = self.p(images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vec = self.m.get_image_features(**inp)
        return self._norm(vec)


class QwenVerifier:
    def __init__(self, model="Qwen/Qwen2.5-VL-7B-Instruct", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.p = None
        self.m = None

    def load(self):
        if self.m is not None:
            return
        dt = torch.float16 if self.dev == "cuda" else torch.float32
        self.m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dt,
            device_map="auto" if self.dev == "cuda" else None,
            attn_implementation="sdpa",
        )
        if self.dev != "cuda":
            self.m.to(self.dev)
        self.p = AutoProcessor.from_pretrained(self.model_name, min_pixels=256 * 28 * 28, max_pixels=640 * 28 * 28)

    def _pick(self, txt):
        a = txt.find("{")
        b = txt.rfind("}")
        raw = txt[a:b + 1] if a >= 0 and b > a else txt
        try:
            res = json.loads(raw)
        except Exception:
            res = {}
        return {
            "match": bool(res.get("match", False)),
            "score": float(res.get("score", 0.0) or 0.0),
            "summary": str(res.get("summary", txt)).strip(),
            "raw": txt.strip(),
        }

    def verify(self, clip, q, meta=""):
        self.load()
        ask = (
            "You are checking whether a CCTV clip matches a natural language query.\n"
            f"query: {q}\n"
            f"tracking_meta: {meta}\n"
            "Return strict JSON only with keys match, score, summary.\n"
            "score must be a number from 0 to 1.\n"
            "Set match true only when the video clearly supports the query.\n"
            "Keep summary short and factual."
        )
        msg = [{
            "role": "user",
            "content": [
                {"type": "video", "path": clip},
                {"type": "text", "text": ask},
            ],
        }]
        inp = self.p.apply_chat_template(
            msg,
            fps=1,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.m.device)
        with torch.no_grad():
            out = self.m.generate(**inp, max_new_tokens=120, do_sample=False)
        ids = [o[len(i):] for i, o in zip(inp.input_ids, out)]
        txt = self.p.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return self._pick(txt)
