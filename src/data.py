"""Data: download Stanford Cars, turn 196 car models into BIG / SMALL labels,
remove duplicates, and split by car model so test cars are models the network never saw."""
import io
import os
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["small", "big"]          # index 1 = "big" = the positive class
BIG_BODY = {"SUV", "Cab", "Van", "Minivan"}          # SUVs, pickup trucks, vans
SMALL_BODY = {"Sedan", "Hatchback", "Coupe", "Convertible", "Wagon"}
# The 15 model names that don't contain a body-type word, labelled by hand:
MANUAL_BODY = {
    "Acura TL Type-S 2008": "Sedan", "Acura Integra Type R 2001": "Coupe",
    "Buick Regal GS 2012": "Sedan", "Chevrolet Corvette ZR1 2012": "Coupe",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007": "Coupe", "Chevrolet HHR SS 2010": "Wagon",
    "Chevrolet Cobalt SS 2010": "Coupe", "Chevrolet TrailBlazer SS 2009": "SUV",
    "Chrysler 300 SRT-8 2010": "Sedan", "Dodge Challenger SRT8 2011": "Coupe",
    "Dodge Charger SRT-8 2009": "Sedan", "FIAT 500 Abarth 2012": "Hatchback",
    "Ford Ranger SuperCab 2011": "Cab", "Jaguar XK XKR 2012": "Coupe",
    "Lamborghini Gallardo LP 570-4 Superleggera 2012": "Coupe",
}
BODY_ORDER = ["SUV", "Cab", "Minivan", "Van", "Sedan", "Hatchback", "Coupe", "Convertible", "Wagon"]

HF = "https://huggingface.co/datasets/tanganke/stanford_cars/resolve/main"
PARQUETS = ["data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet",
            "data/test-00000-of-00002.parquet", "data/test-00001-of-00002.parquet"]
DATA_DIR = os.environ.get("CARS_DATA_DIR", "data/raw")   # Colab: fast local disk, set in the setup cell
IMG_DIR = os.path.join(DATA_DIR, "images")
META = os.path.join(DATA_DIR, "meta.csv")
CACHE_ZIP = "data/cars_resized.zip"                      # resized copy kept in Drive between sessions
SPLIT_DIR = "splits"
SHORT_SIDE = 256                                         # we store images resized to this (faster loading)


def body_type(model_name):
    if model_name in MANUAL_BODY:
        return MANUAL_BODY[model_name]
    for b in BODY_ORDER:
        if re.search(rf"\b{b}\b", model_name):
            return b
    raise ValueError(f"no body type for {model_name}")


def size_label(model_name):
    return "big" if body_type(model_name) in BIG_BODY else "small"


def model_group(model_name):
    """Make + first model word, e.g. 'Chevrolet Silverado 1500 Regular Cab 2012' -> 'Chevrolet Silverado'.
    All years/variants of a car family land in the same group, so they can't be split across train and test."""
    return " ".join(model_name.split()[:2])


def class_names():
    """The 196 model names, in label order, read from the dataset card."""
    text = urllib.request.urlopen(f"{HF}/README.md").read().decode()
    names = re.findall(r"'(\d+)': (.+)", text)
    return [n.strip() for _, n in names][:196]


def download():
    """Get the images onto local disk: from the Drive cache if we have it, else from Hugging Face (~2 GB)."""
    if os.path.exists(META):
        return
    if os.path.exists(CACHE_ZIP):
        print("Unzipping cached images ...")
        with zipfile.ZipFile(CACHE_ZIP) as z:
            z.extractall(DATA_DIR)
        return
    os.makedirs(IMG_DIR, exist_ok=True)
    names = class_names()
    rows = []
    for part in PARQUETS:
        local = os.path.join(DATA_DIR, os.path.basename(part))
        print("Downloading", part, "...")
        urllib.request.urlretrieve(f"{HF}/{part}", local)
        df = pd.read_parquet(local)
        tag = "tr" if "train" in part else "te"
        shard = part.split("-")[1]
        for i, (img, label) in enumerate(zip(df["image"], df["label"])):
            im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            w, h = im.size
            scale = SHORT_SIDE / min(w, h)
            if scale < 1:                                   # only shrink, never enlarge
                im = im.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
            name = f"{tag}{shard}_{i:05d}.jpg"
            im.save(os.path.join(IMG_DIR, name), quality=90)
            rows.append({"path": name, "model_id": int(label), "model": names[label],
                         "orig_w": w, "orig_h": h})
        os.remove(local)
        del df
    meta = pd.DataFrame(rows)
    meta["body"] = meta["model"].map(body_type)
    meta["label"] = meta["model"].map(size_label)
    meta["group"] = meta["model"].map(model_group)
    meta.to_csv(META, index=False)
    print("Saving a resized copy to", CACHE_ZIP, "so the next notebooks start fast ...")
    os.makedirs(os.path.dirname(CACHE_ZIP), exist_ok=True)
    with zipfile.ZipFile(CACHE_ZIP, "w", zipfile.ZIP_STORED) as z:
        z.write(META, "meta.csv")
        for name in meta["path"]:
            z.write(os.path.join(IMG_DIR, name), f"images/{name}")


def load_meta():
    return pd.read_csv(META)


def perceptual_hashes(paths):
    """64-bit perceptual hash per image, packed into one integer each."""
    import imagehash
    out = np.empty(len(paths), dtype=np.uint64)
    for k, p in enumerate(paths):
        bits = imagehash.phash(Image.open(os.path.join(IMG_DIR, p))).hash.flatten()
        out[k] = np.packbits(bits).view(">u8")[0]
    return out


def near_duplicate_pairs(hashes, max_dist=4, chunk=2000):
    """All pairs (i < j) whose hashes differ in <= max_dist bits. Done in chunks to save memory."""
    pairs = []
    for s in range(0, len(hashes), chunk):
        block = hashes[s:s + chunk, None] ^ hashes[None, :]
        dist = np.bitwise_count(block)
        ii, jj = np.where(dist <= max_dist)
        ii = ii + s
        keep = ii < jj
        pairs += list(zip(ii[keep], jj[keep]))
    return pairs


def make_splits(df, seed=42):
    """About 70 / 15 / 15, whole car families (groups) per split, big/small ratio kept similar."""
    from sklearn.model_selection import StratifiedGroupKFold
    df = df.copy()
    outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, df["label"], df["group"]))
    rest = df.iloc[rest_idx]
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_idx, val_idx = next(inner.split(rest, rest["label"], rest["group"]))
    df["split"] = ""
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    df.loc[rest.index[train_idx], "split"] = "train"
    df.loc[rest.index[val_idx], "split"] = "val"
    return df


def save_splits(df):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for s in ["train", "val", "test"]:
        df[df["split"] == s][["path", "label", "model", "body", "group"]].to_csv(f"{SPLIT_DIR}/{s}.csv", index=False)


def load_split(name):
    path = f"{SPLIT_DIR}/{name}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run notebooks/01_data.ipynb first.")
    return pd.read_csv(path)


# ---------- PyTorch side ----------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train):
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.6, 1.8)),
            T.RandomHorizontalFlip(),          # a car facing left is the same size as facing right
            T.ColorJitter(0.3, 0.3, 0.3),      # day / dusk / shade
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),                  # squash, so the whole car stays in view
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CarDataset:
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = [CLASSES.index(l) for l in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(IMG_DIR, self.paths[i])).convert("RGB")
        return self.transform(img), self.labels[i]
