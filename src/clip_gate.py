import numpy as np
import torch
from PIL import Image

from .data import CLASSES

CLIP_NAME = "openai/clip-vit-base-patch16"

CLASS_PROMPTS = {
    "small": ["a photo of a sedan", "a photo of a hatchback", "a photo of a coupe",
              "a photo of a convertible", "a photo of a sports car", "a photo of a station wagon",
              "a photo of a small car"],
    "big": ["a photo of an SUV", "a photo of a pickup truck", "a photo of a van",
            "a photo of a minivan", "a photo of a large off-road vehicle", "a photo of a cargo van",
            "a photo of a big car"],
}
OUT_OF_SCOPE_PROMPTS = [
    "a photo of a person", "a photo of a face", "a photo of an animal", "a photo of a building",
    "a photo of an empty road", "a photo of an empty parking space", "a landscape photo",
    "a photo of a motorcycle", "a photo of a bicycle", "a photo of a bus", "a photo of a semi truck",
    "a photo of a page of text", "a microscope image", "a photo of the night sky",
]


class ClipZeroShot:
    def __init__(self, device=None, model_name=CLIP_NAME):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.prompts, self.owner = [], []
        for k, c in enumerate(CLASSES):
            self.prompts += CLASS_PROMPTS[c]
            self.owner += [k] * len(CLASS_PROMPTS[c])
        self.prompts += OUT_OF_SCOPE_PROMPTS
        self.owner += [-1] * len(OUT_OF_SCOPE_PROMPTS)
        self.owner = np.array(self.owner)

        with torch.no_grad():
            tok = self.processor(text=self.prompts, return_tensors="pt", padding=True).to(self.device)
            t = self._as_tensor(self.model.get_text_features(**tok))
            self.text_emb = t / t.norm(dim=-1, keepdim=True)

    @staticmethod
    def _as_tensor(out):
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def __call__(self, images, batch_size=32):
        class_probs, p_oos = [], []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            pix = self.processor(images=batch, return_tensors="pt").to(self.device)
            e = self._as_tensor(self.model.get_image_features(**pix))
            e = e / e.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * e @ self.text_emb.T
            p = logits.softmax(dim=-1).cpu().numpy()
            per_class = np.stack([p[:, self.owner == k].sum(1) for k in range(len(CLASSES))], 1)
            oos = p[:, self.owner == -1].sum(1)
            class_probs.append(per_class / per_class.sum(1, keepdims=True))
            p_oos.append(oos)
        return np.concatenate(class_probs), np.concatenate(p_oos)


    @torch.no_grad()
    def text_embeddings(self, texts):
        tok = self.processor(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        t = self._as_tensor(self.model.get_text_features(**tok))
        return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def image_embeddings(self, images, batch_size=32):
        out = []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            pix = self.processor(images=batch, return_tensors="pt").to(self.device)
            e = self._as_tensor(self.model.get_image_features(**pix))
            out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)


def load_pil(paths):
    return [Image.open(p).convert("RGB") for p in paths]
