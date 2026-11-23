"""The full system: ResNet-18 + CLIP -> a decision for one photo.

Decision rule (same code is used for evaluation and in app.py):
  1. CLIP says "not a car" (p_out_of_scope >= oos_threshold)  -> "not a car"
  2. p_big = w * resnet_p_big + (1 - w) * clip_p_big
  3. p_big >= big_threshold -> "big", otherwise -> "small"
The big_threshold is chosen on validation so that almost every big car is caught (see notebook 04).
"""
import json
import os

import numpy as np
import torch

from .data import CLASSES, get_transforms
from .model import build_resnet18, get_device

CONFIG_PATH = "results/system_config.json"
WEIGHTS_PATH = "weights/resnet18_finetuned.pt"
BIG = CLASSES.index("big")


def mix(resnet_probs, clip_probs, w):
    return w * resnet_probs + (1 - w) * clip_probs


def decide(resnet_probs, clip_probs, p_oos, w, big_threshold, oos_threshold):
    """Vectorised decisions. Returns (decision strings, p_big)."""
    p_big = mix(resnet_probs, clip_probs, w)[:, BIG]
    decisions = np.where(p_oos >= oos_threshold, "not a car",
                         np.where(p_big >= big_threshold, "big", "small"))
    return decisions.tolist(), p_big


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def save_config(**values):
    cfg = load_config() if os.path.exists(CONFIG_PATH) else {}
    cfg.update(values)
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


class CarSizer:
    """Load once, then call .predict(pil_image)."""

    def __init__(self, weights_path=WEIGHTS_PATH, config=None):
        from .clip_gate import ClipZeroShot
        self.device = get_device()
        self.cfg = config or load_config()
        self.resnet = build_resnet18(freeze_backbone=False, pretrained=False)
        self.resnet.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.resnet.to(self.device).eval()
        self.clip = ClipZeroShot(device=self.device)
        self.tf = get_transforms(train=False)

    @torch.no_grad()
    def predict(self, image):
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        r = torch.softmax(self.resnet(x), 1).cpu().numpy()
        c, oos = self.clip([image])
        d, p_big = decide(r, c, oos, self.cfg["w"], self.cfg["big_threshold"], self.cfg["oos_threshold"])
        return {"decision": d[0], "p_big": float(p_big[0]), "p_not_car": float(oos[0])}
