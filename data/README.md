# Data

**Dataset:** Stanford Cars, by J. Krause, M. Stark, J. Deng and L. Fei-Fei (2013)
- Official page: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Download used: the Hugging Face mirror https://huggingface.co/datasets/tanganke/stanford_cars, files `data/train-0000{0,1}-of-00002.parquet` and `data/test-0000{0,1}-of-00002.parquet` (~2 GB in total). Each row holds the JPEG bytes and the model label (0–195). The 196 model names are read from the dataset card.
- **Licence: non-commercial research use only.** Images were collected from the web and belong to their owners. We use them for coursework only and do not redistribute them.

**Contents:** 16,185 photos of 196 car models (8,144 + 8,041 in the original train/test halves; we merge both and re-split, see below).

**How to get it:** you don't need to do anything. `src/data.py → download()` runs automatically in every notebook and:
1. downloads the 4 parquet files,
2. saves every image resized so its shorter side is 256 px (faster training; small images are never enlarged),
3. writes `meta.csv` (file, model, body type, big/small label, car family, original size),
4. saves the resized copy to `data/cars_resized.zip` (in Google Drive), so the next notebooks unzip it in seconds instead of downloading again.

No images are committed: `*.zip` and `data/raw/` are in `.gitignore`.

**Labels:** body type is taken from the model name (SUV / Cab / Van / Minivan → **big**; Sedan / Hatchback / Coupe / Convertible / Wagon → **small**). The 15 names with no body-type word were labelled by hand in `src/data.py → MANUAL_BODY`.

**Changes we made** (notebook 01): dropped thumbnails (shorter side < 64 px) and near-duplicate photos that appear under two different car families, then split 70/15/15 by car family (`splits/*.csv`).

**Non-car test photos** (notebook 03): sample images bundled with scikit-image (`skimage.data`).
