# Big car or small car? ResNet-18 + CLIP

Classifies a photo of a car as **big** (SUV, pickup truck, van, minivan) or **small** (sedan, hatchback, coupe, convertible, wagon). It also uses CLIP to answer **"not a car"** for photos that aren't cars. It is tested only on **car models the network never saw during training**.

Course project for *Computer Vision for Developers* (SDAIA Academy, Nov 2026). Team: **Mosaad Alrubaish** and **Majed Almutairi**

---

## 1. Use case card

| | |
|---|---|
| **User** | Operator of a multi-storey car park (mall, university, hospital) that has **compact-only levels** with narrow bays and low ceilings, and **large-bay levels** |
| **Input** | One photo of the arriving car from the entrance camera |
| **Output** | **big** → screen directs the driver to the large-bay levels · **small** → any level · **not a car** → no action |
| **Cost of a false negative** (big car called small) | **High.** An SUV or van sent to a compact level can't park, blocks the ramp, or scrapes the low ceiling and neighbouring cars. |
| **Cost of a false positive** (small car called big) | **Low.** A sedan uses a large bay, which wastes some capacity. |
| **Success metric and target** | On the sealed test set of unseen car models: **recall(big) ≥ 95 %** (at most 1 in 20 big cars missed) with **precision(big) ≥ 80 %**. Also report average precision (PR-AUC). |

Because a missed big car is the costly error, the decision threshold is set low on purpose (see §5).

## 2. Data

- **Source:** [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) (Krause et al., 2013), downloaded as parquet files from the Hugging Face mirror [`tanganke/stanford_cars`](https://huggingface.co/datasets/tanganke/stanford_cars) (~2 GB). **Licence: non-commercial research use**, which fits this course project. Details in [`data/README.md`](data/README.md).
- **Size and classes:** 16,185 web photos of 196 car models (make, model, year). Each model name contains its body type (e.g. *BMW X5 **SUV** 2007*), which we map to **big** (SUV, Cab = pickup, Van, Minivan) or **small** (Sedan, Hatchback, Coupe, Convertible, Wagon). 15 names without a body-type word were labelled by hand (`src/data.py → MANUAL_BODY`; full list in `results/model_labels.csv`). Result: **5,525 big / 10,660 small** (about 1 : 2).
- **Split method:** about 70 / 15 / 15 train / val / test, **grouped by car family** (make + model, e.g. every *Chevrolet Silverado* variant and year). A family is only ever in one split, so **every test car is a model the network never trained on**. The big/small ratio is kept similar in each split. The splits are committed in `splits/`.
- **Cleaning (notebook 01):** dropped images whose shorter side is under 64 px (thumbnails), and photos that are near-duplicates (perceptual hash distance ≤ 4) **across different car families**: the same photo filed under two models, which would leak between splits.
- **Known problems:**
  - The label comes from the **body type in the name, not measured size**. A long luxury sedan counts as *small* and a compact SUV as *big*. Wagons are labelled small.
  - **Imbalance** (1/3 big), so we use a class-weighted loss and report precision/recall for *big*.
  - **Web photos** (ads, reviews, car shows): mostly clean side or front views, not an overhead entrance camera.

## 3. Model

_TODO after notebook 02: paste the numbers from your own run._

- **Baseline:** ImageNet ResNet-18 with a **frozen backbone**; only a new 2-class head is trained.
- **Trained model:** the same ResNet-18 with **all layers fine-tuned**. This is the *only* difference from the baseline: same data, augmentation (random crop, horizontal flip, colour jitter), class-weighted cross-entropy, AdamW with cosine schedule, 8 epochs, seed 42. We keep the epoch with the best **validation macro-F1**.
- **Key settings:** input 224×224 · batch 64 · lr 1e-3 (frozen) / 2e-4 (fine-tuned) · weight decay 1e-4
- **Curves:** `results/curves.png`
- **Result (validation macro-F1):** baseline _x.xxx_ → fine-tuned _x.xxx_

## 4. Foundation component: CLIP (ViT-B/16)

A 2-class ResNet **must** answer big or small, even for a person, a bus or an empty bay. CLIP has learned from 400M image–text pairs. It knows the words *SUV, pickup truck, van, sedan* for car models it has seen on the web, including the ones missing from our training set. It is built into the decision rule (`src/system.py`), not run as a separate demo:

1. **Not-a-car gate:** CLIP compares the photo with car prompts and non-car prompts ("a photo of a person", "an empty parking space", "a bus", "a motorcycle", …). If p(not a car) ≥ threshold, the answer is **not a car**. The threshold is the lowest value that rejects ≤ 1 % of validation car photos.
2. **Second opinion:** `p_big = w · ResNet + (1 − w) · CLIP`, with `w` chosen on validation by log-loss.

_TODO after notebook 03:_ chosen `w` = _x.xx_, gate threshold = _x.xx_. Non-car photos rejected: _x / 18_ (`results/not_a_car_gate.png`, `results/mixing_weight.png`).

## 5. Evaluation (sealed test set: unseen car models)

_TODO after notebook 04: paste the table from its last cell._

| model | accuracy @0.5 | F1(big) @0.5 | recall(big) @0.5 | avg precision |
|---|---|---|---|---|
| CLIP zero-shot (no training) | | | | |
| ResNet-18 frozen (baseline) | | | | |
| ResNet-18 fine-tuned | | | | |
| Fine-tuned + CLIP mix | | | | |

- **PR curve:** `results/pr_curve.png` · **Confusion matrix:** `results/confusion_matrix.png` · **Accuracy by body type:** `results/accuracy_by_body_type.csv`
- **Threshold:** _0.xx_. We chose it on **validation** as the highest p(big) threshold that keeps recall(big) ≥ 95 %. A missed big car is the expensive error, so we accept extra small cars being sent to the large-bay levels.
- **Full system on test:** recall(big) _xx %_ · precision(big) _xx %_ · big cars missed _n_ · small cars sent to big levels _n_ · real cars called "not a car" _x %_

## 6. Failure analysis

_TODO after notebook 04: pick 5+ images from `results/failures/` (listed in `failures.csv`)._

| # | Image | Car model | Mistake | Likely cause | Fix tried / would try |
|---|---|---|---|---|---|
| 1 | `results/failures/01_...jpg` | | missed big | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

## 7. Limitations

- "Big" means **body type**, not real dimensions. A real car park would care about height and length. It would need a size measurement (calibrated camera or sensors) or a per-model size lookup.
- Trained on **web photos**: clean, well-lit, mostly side and front views. An entrance camera sees cars from above, at night, in rain and partly hidden by barriers, so accuracy would drop. Production would need footage from the real camera to fine-tune and test.
- Car models from **2012 and earlier** only. Newer designs (crossovers that look between SUV and hatchback) are exactly the hard cases.
- Only **one car per photo** is assumed. Production would add a car detector first.
- The test set has a limited number of car families, so a few hard models can move the numbers noticeably.

## 8. How to run

From a fresh Google Colab (GPU runtime: *Runtime → Change runtime type → T4 GPU*):

1. Open `notebooks/01_data.ipynb` in Colab (*File → Open notebook → GitHub*, paste this repo's URL).
2. Run the first cell. It mounts Google Drive, clones the repo to `MyDrive/car-size` (shared by all four notebooks), installs `requirements.txt`, and keeps the images on Colab's fast local disk.
3. Run the notebooks in order:
   - `01_data` downloads ~2 GB from Hugging Face (3–5 min) and saves a resized copy to Drive (`data/cars_resized.zip`), so later notebooks start in seconds.
   - Then `02_train` (~15–25 min on a T4) → `03_foundation` → `04_evaluate`.
4. Results appear in `results/`, and the models are saved in `weights/`.

Run the demo locally: `pip install -r requirements.txt && python app.py`

**Deploy (Hugging Face Spaces):** create a Space with the **Gradio** SDK and upload `app.py`, `requirements.txt`, `src/`, `weights/resnet18_finetuned.pt` and `results/system_config.json`. Live demo: _link_

## Repo layout

```
notebooks/   01_data · 02_train · 03_foundation · 04_evaluate
src/         data.py (download, big/small labels, dedup, split by car family)
             model.py (ResNet-18 + training loop) · clip_gate.py (CLIP zero-shot + not-a-car gate)
             system.py (full decision rule)
splits/      train/val/test CSVs (created by 01)
results/     curves, PR curve, confusion matrix, failures/
weights/     trained models (created by 02)
app.py       Gradio demo
```

Dataset credit: J. Krause, M. Stark, J. Deng, L. Fei-Fei. *3D Object Representations for Fine-Grained Categorization*, 4th IEEE Workshop on 3D Representation and Recognition (3dRR-13), 2013.
