import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


class EventTagger:
    def __init__(self, model="microsoft/xclip-base-patch32", labels=None, device=None):
        self.model_name = model
        self.labels = labels or [
            "a person sitting",
            "a person standing",
            "a person walking",
            "a person running",
            "people fighting",
            "a person falling",
            "a traffic collision",
            "vehicles moving in traffic",
            "a crowd gathering",
            "a person loitering",
        ]
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

    def _sample_frames(self, frames, want=8):
        if not frames:
            return []
        if len(frames) >= want:
            idx = np.linspace(0, len(frames) - 1, want).round().astype(int).tolist()
            return [frames[i] for i in idx]
        out = list(frames)
        while len(out) < want:
            out.append(out[-1])
        return out

    def score(self, frames, top_k=2, min_score=0.18):
        frames = self._sample_frames(frames)
        if not frames:
            return []
        self.load()
        inp = self.p(text=self.labels, videos=frames, return_tensors="pt", padding=True).to(self.dev)
        with torch.no_grad():
            out = self.m(**inp)
        probs = out.logits_per_video.softmax(dim=-1)[0].detach().cpu().tolist()
        rows = sorted(zip(self.labels, probs), key=lambda x: x[1], reverse=True)
        keep = []
        for label, score in rows[:top_k]:
            if score >= min_score:
                keep.append({"label": label, "score": float(score)})
        return keep
