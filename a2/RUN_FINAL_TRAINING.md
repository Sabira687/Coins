# Running the final training (Assignment 2)

Trains DINOv2 ViT-S/14 on all training data and writes everything the app needs to `results_a2/`.
Nothing here is started automatically.

## Split
| Subset | Content |
|---|---|
| Train | Part 1 (`c0`, 6 setups) + Part 2 coins `c1`–`c6` of every denomination |
| Validation | Part 2 coin `c7` of every denomination (chooses the best epoch) |
| Test | Part 2 coin `c8` of every denomination (never used for training or selection) |

The model is trained on 31 of 41 coins, so **its numbers must not go into the paper** (paper experiments live in `paper_experiments.ipynb`).

## Expected cost
About 2300 training images. From the logged paper runs (1109 images: 5–11 min) estimate **20–25 minutes** on the Apple GPU (`mps`). DINOv2 runs in micro-batches of 8 with gradient accumulation over 4, so memory is not a problem.

## Steps
1. Plug in the charger, close heavy apps (the GPU is busy; the interface may lag).
2. Pipeline check, about 1 minute: in `a2_final_training.ipynb` set `FAST = True` in the second cell, run all cells, then set it back to `False` and delete `results_a2_fast/`.
3. Real run, either in Jupyter (`Run All`) or from the terminal:

```bash
source .venv/bin/activate
caffeinate -i jupyter nbconvert --to notebook --execute --inplace a2_final_training.ipynb --ExecutePreprocessor.timeout=-1
```

4. Progress: `results_a2/log.txt` (one line per epoch).
5. If the run is interrupted, run the same command again: it continues from `results_a2/ckpt.pt` (saved after every epoch).

## Outputs (`results_a2/`)
| File | Content |
|---|---|
| `final_dinov2.pt` | weights of the best epoch |
| `index.npz` | embeddings, paths, labels of the training images (nearest-neighbour gallery in the app) |
| `final_metrics.json` | test accuracy and macro F1 (linear head and kNN, k = 1, 5, 10), latency on this machine |
| `history.json`, `final_curves_confusion.png` | training curves and test confusion matrix |
| `ckpt.pt`, `log.txt` | resume state (git-ignored) and log |

## Code layout
`coinlib/` holds the shared code (data, DINOv2 model, resumable training, coin cropping from `extract_frames_p2.py`). The app will import the same package. `paper_experiments.ipynb` is unchanged.
