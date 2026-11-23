"""CLIP (foundation model) used two ways inside the system:
1. Out-of-scope gate: is this even a car? (ResNet can only answer "big" or "small".)
2. Second opinion: zero-shot big/small probabilities that are mixed with ResNet's."""
import numpy as np
import torch
from PIL import Image

from .data import CLASSES

CLIP_NAME = "openai/clip-vit-base-patch16"

# Several prompts per class; CLIP's score for a class is the total over its prompts.
CLASS_PROMPTS = {
    "small": ["a photo of a sedan", "a photo of a hatchback", "a photo of a coupe",
              "a photo of a convertible", "a photo of a sports car", "a photo of a station wagon",
              "a photo of a small car"],
    "big": ["a photo of an SUV", "a photo of a pickup truck", "a photo of a van",
            "a photo of a minivan", "a photo of a large off-road vehicle", "a photo of a cargo van",
            "a photo of a big car"],
}
# Things that are NOT a car. If these win, the system answers "not a car" instead of guessing.
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

        self.prompts, self.owner = [], []  # owner = class index, or -1 for out-of-scope
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
        # some transformers versions return an output object instead of a plain tensor
        return out if torch.is_tensor(out) else out.pooler_output

    @torch.no_grad()
    def __call__(self, images, batch_size=32):
        """images: list of PIL images.
        Returns (class_probs [N, 2] that sum to 1, p_out_of_scope [N])."""
        class_probs, p_oos = [], []
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i:i + batch_size]]
            pix = self.processor(images=batch, return_tensors="pt").to(self.device)
            e = self._as_tensor(self.model.get_image_features(**pix))
            e = e / e.norm(dim=-1, keepdim=True)
            logits = self.model.logit_scale.exp() * e @ self.text_emb.T
            p = logits.softmax(dim=-1).cpu().numpy()          # over every prompt
            per_class = np.stack([p[:, self.owner == k].sum(1) for k in range(len(CLASSES))], 1)
            oos = p[:, self.owner == -1].sum(1)
            class_probs.append(per_class / per_class.sum(1, keepdims=True))  # renormalise over small/big
            p_oos.append(oos)
        return np.concatenate(class_probs), np.concatenate(p_oos)


def load_pil(paths):
    return [Image.open(p).convert("RGB") for p in paths]
