# Denomination or Specimen? Kazakhstan Tenge Coin Classification

Kurbankhozha Sabira · Astana IT University, Software Engineering (SE-2424) · Computer Vision course project

This repository contains a two-part image dataset of Kazakhstan tenge coins (5, 10, 20, 50 and 100 tenge), the code used to build it from smartphone videos, and the experiments for a paper prepared for *ELCVIA – Electronic Letters on Computer Vision and Image Analysis*.

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
| **Part 2** | Specimen: 5 coins per denomination (`c1`–`c5`), 25 coins in total | Conditions: one white background, constant lighting |

The Part 1 coins are not among the Part 2 coins. Conditions in Part 2 are fixed on purpose, so that any change in accuracy can be attributed to the specimen alone; robustness to conditions is tested in Part 1.

### Experiment 1 — new conditions, same coin

Part 1 only. Training on setups 2, 4, 5 and 6, validation on setup 3, test on setup 1 (white paper, daylight). No daylight images are used for training. Three seeds per model.

### Experiment 2 — new coins, same conditions

Leave-one-specimen-out over the Part 2 coins (5 folds). In fold *k*, coin *k* of every denomination is the test set, coin *k + 1* is the validation set, and the remaining three coins form the Part 2 training set.

On the same test coins we compare two versions of every model:

- **P1-trained** — coin `c0` in all six setups (including the white background), randomly subsampled to the same number of images per class as the Part 2 training set, evenly across setups and coin sides;
- **P2-trained** — three Part 2 coins on the white background.

Both versions select their best epoch on the same validation coin. The test coins are new to both, the test condition is familiar to both, and the training sets have the same size, so the difference between them reflects only what they were trained on: one coin under many conditions or several coins under one condition.

The **specimen gap** is the accuracy of a P1-trained model in Experiment 1 minus its accuracy on unseen coins in Experiment 2.

## Models

| Model | Pretraining | Parameters | Training |
|---|---|---|---|
| Simple CNN | none (from scratch) | 390,661 | Adam, lr 1e-3, up to 30 epochs |
| ResNet18 | ImageNet (supervised) | 11,179,077 | head only for 5 epochs, then full fine-tuning (AdamW, lr 1e-4, up to 15 epochs) |
| DINOv2 ViT-S/14 | LVD-142M (self-supervised) | ~21M | head only for 5 epochs, then full fine-tuning (AdamW, lr 1e-5 backbone / 1e-4 head, up to 15 epochs) |

All models use 224×224 inputs with ImageNet normalisation, cross-entropy with label smoothing 0.05, batch size 32, best-epoch selection on validation loss and early stopping. Training augmentation: rotation 0–360°, random resized crop (60–100%), random perspective, brightness/contrast jitter, weak saturation and hue jitter, Gaussian blur; no flips, because a mirrored coin never occurs in reality.

Evaluation: accuracy and macro-F1 (mean ± std over seeds or folds), confusion matrices, per-coin and per-side accuracy, Grad-CAM (ResNet18) and nearest-neighbour retrieval in the DINOv2 embedding space.

## Dataset

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

### Part 2 counts

987 images, about 40 per coin: 5 tenge — 192, 10 tenge — 200, 20 tenge — 200, 50 tenge — 200, 100 tenge — 195.

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
├── extract_frames_p2.py      # Part 2: videos -> cropped frames
├── paper_experiments.ipynb   # Experiments 1 and 2, tables and figures for the paper
└── results_paper/            # tables (.csv), figures and per-run predictions
```

## Reproducing the experiments

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision opencv-python numpy scikit-learn matplotlib pandas jupyter
jupyter notebook paper_experiments.ipynb
```

Set `FAST = True` in the first code cell for a quick pipeline check, then `FAST = False` for the full run (39 training runs). Finished runs are saved immediately and skipped on re-run, so the notebook can be interrupted and resumed. DINOv2 is downloaded through `torch.hub` on first use.

To rebuild the datasets from your own videos, place them in `raw_videos/<class>/<side><setup>.mp4` (Part 1, e.g. `A3.mp4`) or `raw_videos_p2/<class>/<coin><side>.mov` (Part 2, e.g. `3R.mov`) and run the corresponding `extract_frames` script.

## Results

To be added after the full run.

## Limitations

- Part 2 uses a single background by design; robustness to conditions is measured on one specimen only.
- The two parts were recorded on different days, by different people and with different phones.
- Only regular coin issues are included; commemorative 50 and 100 tenge coins are not.
- Frames extracted from a video are correlated, so the effective dataset size is smaller than the image count.
- Cropping relative to the coin's own radius removes absolute coin size, the main physical cue separating 20 and 50 tenge.
- Five specimens per denomination is still a small number.

## Citation

Paper in preparation. Citation details will be added after submission.
