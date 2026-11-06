# Denomination or Specimen? Kazakhstan Tenge Coin Classification

Kurbankhozha Sabira, [TODO: co-author] · Astana IT University, Software Engineering (SE-2424) · Computer Vision course project

This repository contains a two-part image dataset of Kazakhstan tenge coins (5, 10, 20, 50 and 100 tenge), the code used to build it from smartphone videos, the experiments for a paper prepared for *ELCVIA – Electronic Letters on Computer Vision and Image Analysis*, and the Assignment 2 deliverables: a phone web app with explanations (**Tenge Scan**) and a report.

| Part | Where |
|---|---|
| Assignment 1 (baselines: HOG+SVM, simple CNN, ResNet18) | `assignment1/` |
| Paper experiments (specimen vs. denomination) | `paper_experiments.ipynb`, `results_paper/` |
| Assignment 2: final DINOv2 model | `a2_final_training.ipynb`, `results_a2/`, `a2/RUN_FINAL_TRAINING.md` |
| Assignment 2: phone app | `app/` (start with `app/README.md`) |
| Assignment 2: report (Russian draft) | `a2/report/Assignment2_report_ru.docx` |

## Research question

When an image dataset is built from very few physical objects, does a classifier learn the class or the individual objects it was trained on?

In a preliminary study with one coin per denomination, a fine-tuned ResNet18 reached 0.969 test accuracy under lighting it had not seen during training. Two observations made us doubt that it had learned the denominations themselves:

- accuracy on the reverse side was the same as on the obverse (0.968 vs 0.969), although the 10, 20 and 50 tenge coins share the same state emblem on the reverse;
- all errors involved the 20 tenge coin, which has the same colour and design as 50 tenge and differs from it mainly in diameter.

The model may have partly memorised specimen-specific details such as wear, scratches and tint. This project measures that effect directly.

## Study design

Each part of the dataset varies one factor and keeps the other fixed.

| | What varies | What is fixed |
|---|---|---|
| **Part 1** | Recording conditions: 6 background × lighting setups | Specimen: 1 coin per denomination (`c0`) |
| **Part 2** | Specimen: 8 coins per denomination (`c1`–`c8`), 40 coins in total | Conditions: one white background, constant lighting, one phone |

The Part 1 coins are not among the Part 2 coins. Conditions in Part 2 are fixed on purpose, so that any change in accuracy can be attributed to the specimen alone; robustness to conditions is tested in Part 1.

### Experiment 1 — new conditions, same coin

Part 1 only. Training on setups 2, 4, 5 and 6, validation on setup 3, test on setup 1 (white paper, daylight). No daylight images are used for training. Three seeds per model.

### Experiment 2 — new coins, same conditions

Leave-one-specimen-out over the Part 2 coins (8 folds). In fold *k*, coin *k* of every denomination is the test set, coin *k + 1* is the validation set, and the remaining six coins form the Part 2 training set.

On the same test coins we compare two versions of every model:

- **P1-trained** — coin `c0` in all six setups, randomly subsampled to the same number of images per class as the Part 2 training set, evenly across setups and coin sides;
- **P2-trained** — six Part 2 coins on the white background.

Both versions select their best epoch on the same validation coin. The test coins are new to both, the test condition is familiar to both, and the training sets have the same size, so the difference between them reflects only what they were trained on: one coin under many conditions or several coins under one condition.

The **specimen gap** is the accuracy of a P1-trained model in Experiment 1 minus its accuracy on unseen coins in Experiment 2.

### Ablation (Assignment 2)

DINOv2 with a frozen backbone (linear head only) against full fine-tuning, on the Experiment 1 split.

## Models

| Model | Pretraining | Parameters | Training |
|---|---|---|---|
| Simple CNN | none (from scratch) | 390,661 | Adam, lr 1e-3, up to 30 epochs |
| ResNet18 | ImageNet (supervised) | 11,179,077 | head only for 5 epochs, then full fine-tuning (AdamW, lr 1e-4, up to 15 epochs) |
| DINOv2 ViT-S/14 | LVD-142M (self-supervised) | ~22M | head only for 5 epochs, then full fine-tuning (AdamW, lr 1e-5 backbone / 1e-4 head, weight decay 0.05, up to 15 epochs) |

All models use 224×224 inputs with ImageNet normalisation, cross-entropy with label smoothing 0.05, effective batch size 32, best-epoch selection on validation loss and early stopping. DINOv2 runs in micro-batches of 8 with gradient accumulation over 4: a batch of 32 does not fit into the GPU memory of a laptop Mac. Training augmentation: rotation 0–360°, random resized crop (60–100%), random perspective, brightness/contrast jitter, weak saturation and hue jitter, Gaussian blur; no flips, because a mirrored coin never occurs in reality.

Evaluation: accuracy and macro-F1 (mean ± std over seeds or folds), confusion matrices, per-coin and per-side accuracy, Grad-CAM (ResNet18) and nearest-neighbour retrieval in the DINOv2 embedding space.

## Dataset

2694 images, 41 coins: Part 1 — 1109 images (one coin), Part 2 — 1585 images (40 coins). No synthetic or generated images.

### Collection pipeline

1. Each coin side is recorded as a short smartphone video while the camera tilts, moves closer and further, and the coin is turned.
2. Each video is split into 20 equal time windows and the sharpest frame of each window is kept (variance of the Laplacian); frames below 0.35 of the video's median sharpness are dropped.
3. The coin is located by Hough circle detection; if that fails, by fitting an ellipse to contours from brightness and saturation edges (tilted coins, palm background); if that fails too, its position is taken from the nearest frame of the same video.
4. A square crop centred on the coin is saved at 512×512. Its half-side is the coin radius times a random factor in [1.3, 3.0], which keeps variation in apparent coin size.
5. All crops were checked manually.

Splits are always made by video or by coin, never by individual frame, because neighbouring frames of a video are nearly identical.

### Part 1 setups

| Setup | Background | Lighting | Role in Experiment 1 |
|---|---|---|---|
| 1 | White paper | Daylight | test |
| 2 | Dark fabric | Lamp | train |
| 3 | Wooden cutting board | Daylight | validation |
| 4 | Palm of a hand | Lamp | train |
| 5 | White paper | Warm side lamp | train |
| 6 | Wooden cutting board | Phone flash | train |

### Image counts

| Class | Part 1 | Part 2 | Total |
|---|---|---|---|
| 5 tenge | 220 | 312 | 532 |
| 10 tenge | 223 | 319 | 542 |
| 20 tenge | 217 | 320 | 537 |
| 50 tenge | 214 | 319 | 533 |
| 100 tenge | 235 | 315 | 550 |
| **Total** | **1109** | **1585** | **2694** |

Per-setup and per-coin counts: `results_paper/p1_counts.csv`, `results_paper/p2_counts.csv`.

### File naming

| | Path |
|---|---|
| Part 1 | `dataset/<split>/<class>/<class>_<A\|R><setup>_<k>.jpg` |
| Part 2 | `dataset_p2/<class>/<class>_c<coin><A\|R>_<k>.jpg` |

`A` is the obverse (numeral), `R` the reverse (emblem). Raw videos are not included in the repository.

## Repository structure

```
├── assignment1/              # Assignment 1: notebook and results (Part 1 only, incl. HOG + SVM baseline)
├── dataset/                  # Part 1 images, split into train / val / test
├── dataset_p2/               # Part 2 images, one folder per denomination
├── extract_frames.py         # Part 1: videos -> cropped frames
├── extract_frames_p2.py      # Part 2: videos -> cropped frames (also the coin detector used by the app)
├── paper_experiments.ipynb   # Experiments 1 and 2, ablation, tables and figures
├── results_paper/            # tables (.csv), figures, per-run predictions, fold-1 weights
├── coinlib/                  # shared code: data, DINOv2, resumable training, detection, predictor, squared-paper size
├── a2_final_training.ipynb   # final DINOv2 model (train: P1 + c1–c6, validation: c7, test: c8)
├── results_a2/               # final weights, embedding index, metrics, curves
├── results_a2_dev/           # development index (fold-1 weights) used when no final model exists
├── app/                      # Tenge Scan: FastAPI server + canvas front end (see app/README.md)
├── size_test/                # photos of coins on squared paper and the diameter measurement
├── a2/                       # Assignment 2: report (a2/report/), run guide for the final training
├── related_work.md           # Related Work section of the paper
└── PAPER_STRUCTURE.md        # paper outline
```

## Reproducing the experiments

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python numpy scikit-learn matplotlib pandas jupyter
jupyter notebook paper_experiments.ipynb
```

Set `FAST = True` in the first code cell for a quick pipeline check, then `FAST = False` for the full run. Finished runs are saved immediately and skipped on re-run, so the notebook can be interrupted and resumed. DINOv2 is downloaded through `torch.hub` on first use.

To rebuild the datasets from your own videos, place them in `raw_videos/<class>/<side><setup>.mp4` (Part 1, e.g. `A3.mp4`) or `raw_videos_p2/<class>/<coin><side>.mov` (Part 2, e.g. `3R.mov`) and run the corresponding `extract_frames` script.

**Final model.** `a2_final_training.ipynb` (about 20–25 minutes on an Apple GPU, resumable); instructions in `a2/RUN_FINAL_TRAINING.md`. The final weights have seen 31 of the 41 coins, so they must not be used for paper numbers.

**App.**

```bash
pip install -r app/requirements.txt
python app/server.py        # prints a QR code and an https address for the phone
```

Details, the demo mode without a camera and the explanation methods are in `app/README.md`.

**Size from squared paper.** `python size_test/measure.py` writes `size_test/diameters.csv` (coin diameters in notebook cells, 1 cell = 5 mm).

## Results

Exact values: `results_paper/*.csv`, figures: `results_paper/figures/`.

### Experiment 1 — new conditions, same coin (accuracy, macro-F1, mean ± std over 3 seeds)

| Model | Accuracy | Macro-F1 | Training, min |
|---|---|---|---|
| Simple CNN | 0.302 ± 0.039 | 0.223 ± 0.042 | 2.7 |
| ResNet18 | 0.965 ± 0.005 | 0.966 ± 0.004 | 2.6 |
| DINOv2 ViT-S/14, frozen backbone | 0.872 ± 0.020 | 0.871 ± 0.021 | 3.2 |
| **DINOv2 ViT-S/14, fine-tuned** | **0.995 ± 0.004** | **0.995 ± 0.004** | 8.8 |

### Experiment 2 — new coins (accuracy, mean ± std over 8 folds)

| Model | P1-trained | P2-trained |
|---|---|---|
| Simple CNN | 0.308 ± 0.038 | 0.687 ± 0.115 |
| ResNet18 | 0.413 ± 0.092 | 0.710 ± 0.114 |
| DINOv2 ViT-S/14 | 0.442 ± 0.109 | 0.768 ± 0.115 |

Specimen gap (Experiment 1 minus Experiment 2, P1-trained): ResNet18 55 points, DINOv2 55 points, simple CNN about 0 (it does not learn in Experiment 1).

Main findings: models trained on one coin are near-perfect on that coin under new light but drop to 0.41–0.44 on other coins; with equal data, six coins in one setup generalise to new coins far better than one coin in six setups; the confusions are systematic (5↔10 and 20↔50), while the bimetallic 100 tenge is almost perfect.

### Final model (Assignment 2)

Trained on Part 1 and coins `c1`–`c6`, validated on `c7`, tested on `c8` (one coin per denomination, 199 images): accuracy 0.834, macro-F1 0.830 (kNN on embeddings, k = 10: 0.839). This is a rough estimate from five coins.

External check on 7 phone photos of five other coins on squared paper (31 coins analysed): 29 / 31 correct without size, 31 / 31 with size. The mean diameters used for the size cue were measured on the same photos, so the second number is optimistic.

| Denomination | 5 | 20 | 10 | 50 | 100 |
|---|---|---|---|---|---|
| Diameter, cells (1 cell = 5 mm) | 3.59 ± 0.02 | 3.80 ± 0.04 | 3.96 ± 0.05 | 4.70 ± 0.04 | 5.25 ± 0.06 |

## Limitations

- Part 2 uses a single background by design; robustness to conditions is measured on one specimen only.
- The two parts were recorded on different days, by different people and with different phones.
- Only regular coin issues are included; commemorative 50 and 100 tenge coins are not.
- Frames extracted from a video are correlated, so the effective dataset size is smaller than the image count.
- Eight specimens per denomination is still a small number; the test of the final model uses five coins.
- Cropping relative to the coin's own radius removes absolute coin size. The app recovers it only when squared paper is visible; the diameters come from one specimen per denomination and were not checked against the official catalogue.
- The app does not check authenticity and must not be used for that or for financial decisions. It stores no images.

## Citation

Paper in preparation. Citation details will be added after submission.
