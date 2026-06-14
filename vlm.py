import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class SearchEncoder:
    def __init__(self, model="openai/clip-vit-base-patch32", device=None):
        self.model_name = model
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.p = None
        self.m = None

    def load(self):
        if self.m is not None:
            return
        self.p = CLIPProcessor.from_pretrained(self.model_name)
        self.m = CLIPModel.from_pretrained(self.model_name)
        self.m.to(self.dev)
        self.m.eval()

    def _norm(self, x):
        x = x.detach().cpu().numpy()[0]
        n = np.linalg.norm(x)
        if n == 0:
            return x.astype(np.float32)
        return (x / n).astype(np.float32)

    def embed_text(self, txt):
        self.load()
        inp = self.p(text=[txt], return_tensors="pt", padding=True).to(self.dev)
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
