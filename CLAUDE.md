# Project context for Claude Code

Owner: Kurbankhozha Sabira (SE-2424, Astana IT University, Software Engineering).
Talk to Sabira in **Russian**. Code, notebooks, README and paper text are in **English**.

This project grew out of a long chat; everything needed to continue is below.

## 1. What this project is

Kazakhstan tenge coin classification (5, 10, 20, 50, 100 tenge) for the Computer Vision course. Three deliverables share one codebase:

| Deliverable | Status | Deadline |
|---|---|---|
| Assignment 1 (one coin per denomination, 6 setups; HOG+SVM, simple CNN, ResNet18) | submitted | done |
| **Assignment 2** (SOTA method + app + report + demo video + repo) | in progress, **current priority** | **7 Nov 2026, 23:59** |
| **Paper** for ELCVIA (instead of endterm + final exams), co-authored with a partner who recorded the Part 2 videos | results collected, text partly written | submit by **10 Nov 2026**; 5-min class presentation in week 10 |

The teacher approved the paper's methodology and literature review, and knows about the 3 extra coins per denomination.

## 2. Research question and design (paper)

Question: when a dataset is built from very few physical objects, does a model learn the class (denomination) or the individual specimens it was trained on?

Two parts, each varies one factor:
- **Part 1 (P1)**: one coin per denomination (`c0`), 6 background × lighting setups. Varies conditions.
- **Part 2 (P2)**: 8 coins per denomination (`c1`–`c8`, 40 coins), one fixed setup: same place, white background, same phone. Varies the specimen. The P1 coin is not among them.

Experiments (all in `paper_experiments.ipynb`):
- **Experiment 1** — new conditions, same coin. P1: train setups 2, 4, 5, 6; val 3; test 1 (paper + daylight). Seeds 42, 43, 44.
- **Experiment 2** — new coins, same conditions. Leave-one-specimen-out, 8 folds: test coin k, val coin k+1, train the other 6 P2 coins. On the same test coins compare P1-trained (c0 in 6 setups) vs P2-trained (6 coins, 1 setup). Both training sets cut to the same number of images per class (evenly over setups/coins and sides); both use the same validation coin. Seed 42.
- **Specimen gap** = Experiment 1 accuracy − Experiment 2 accuracy of the P1-trained model.
- **Ablation (Assignment 2 only, not in the paper)**: DINOv2 frozen backbone vs fully fine-tuned, Experiment 1 split.

Models (same in both parts): Simple CNN (from scratch, 390,661 params), ResNet18 (ImageNet, two-stage fine-tuning, 11,179,077 params), DINOv2 ViT-S/14 (self-supervised, two-stage fine-tuning). HOG+SVM was dropped (0.218 in Assignment 1).

## 3. Results so far (approximate, read exact values from `results_paper/*.csv`)

| Model | Exp. 1 (same coin, new light) | Exp. 2 P1-trained | Exp. 2 P2-trained |
|---|---|---|---|
| Simple CNN | ~0.30 | 0.31 | 0.69 |
| ResNet18 | ~0.97 | 0.41 | 0.71 |
| DINOv2 | ~1.00 | 0.44 | 0.77 |

(Exp. 2 numbers above are pooled over folds from the confusion matrices; per-fold mean ± std is in the CSVs.)

Findings:
1. Models trained on one coin are near-perfect on that coin under new light but drop to 0.41–0.44 on other coins: specimen gap ≈ 55 points.
2. With equal data, six coins in one setup generalise to new coins far better than one coin in six setups.
3. DINOv2 is best in both experiments but has the same specimen gap as ResNet18.
4. The simple CNN fails in Exp. 1 (0.30) but reaches 0.69 when trained on six coins — the pretraining advantage shrinks when specimens vary.
5. P2 confusions are systematic: 5 ↔ 10 and 20 ↔ 50 (similar colour); 100 tenge (bimetallic) is almost perfect.
6. Hard specimens for DINOv2 P2: 10 tenge c4 and c7, 20 tenge c5, 50 tenge c6 and c7. To be explained with the coin table (issue year, condition) — pending from the co-author.
7. Grad-CAM: P1-trained ResNet18 partly attends to background on errors; P2-trained attends to the coin centre.
8. DINOv2 nearest neighbours of misclassified 5-tenge coins are mostly 5-tenge coins although the linear head predicts 10/100 → the embedding holds the right information; kNN on embeddings might do better.

Figures: `results_paper/figures/` (`e1_curves.png`, `e2_accuracy.png`, `e2_confusion.png`, `e2_per_coin.png`, `e2_curves_fold1.png`, `gradcam_p1_vs_p2.png`, `dinov2_neighbours.png`, `p2_specimens.png`).

## 4. Repository layout

```
assignment1/              Assignment 1 notebook + results (do not change)
dataset/                  P1 images: <split>/<class>/<class>_<A|R><setup>_<k>.jpg
dataset_p2/               P2 images: <class>/<class>_c<coin><A|R>_<k>.jpg
extract_frames.py         P1 videos -> frames (raw_videos/<class>/<A|R><setup>.mp4)
extract_frames_p2.py      P2 videos -> frames (raw_videos_p2/<class>/<coin><A|R>.mov), any coin number
paper_experiments.ipynb   all experiments, tables, figures (resumable: one .npz per run in results_paper/runs/)
results_paper/            CSV tables, figures, runs/, weights/ (fold-1 weights of ResNet18/DINOv2), log.txt
README.md                 repo README (needs update: 8 coins, co-author, app, results)
PAPER_STRUCTURE.md        paper outline
related_work.md           finished Related Work section + verified reference list
```
`A` = obverse (numeral), `R` = reverse (emblem). Raw videos, `.venv`, `.idea`, `*.pt` are git-ignored. Remote: GitHub, public repo.

## 5. Environment — important pitfalls

- macOS, Apple Silicon, Homebrew Python 3.14, virtual env `.venv` (`source .venv/bin/activate`). Homebrew Python refuses global pip installs — always use the venv.
- torch 2.14.1, device `mps`. OpenCV 5.0.
- **DINOv2 on MPS**: a batch of 32 does not fit in GPU memory → 136 s/step. Fixed by micro-batches of 8 with gradient accumulation over 4 (0.36 s per micro-step). Keep this for any DINOv2 training/inference.
- DINOv2 is loaded with `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True, skip_validation=True)` (without `skip_validation` GitHub's API rate limit breaks loading). Its positional embedding is resized once to the 16×16 grid for 224×224 inputs (see `dino_backbone()` in the notebook).
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in the notebook.
- Preprocessing: 224×224, ImageNet mean/std. Training augmentation: rotation 0–360°, RandomResizedCrop(0.6–1.0), perspective p=0.3, brightness/contrast ±0.3, saturation ±0.15, hue ±0.02, blur p=0.2, **no flips**.
- Coin cropping: Hough circle → ellipse fit on brightness+saturation edges → nearest frame; square crop with random margin 1.3–3.0 radii, saved 512×512.

## 6. Assignment 2 — what is left (current priority)

Requirements: same topic/dataset (may extend), 1–2 SOTA methods compared to the Assignment 1 baseline, an app (upload/camera → prediction → visual explanation, showing inference time and device; local run guide is acceptable), report 6–8 pages, demo video 3–5 min, repo, defence in practice class.

Already done: extended dataset (P2), DINOv2 vs CNN/ResNet18 on the Assignment 1 split (Exp. 1), metrics, curves, ablation (`table_ablation.csv`), Grad-CAM, nearest neighbours.

To do, in order:
1. **Final app model**: fine-tune DINOv2 on all data (P1 all setups + all 40 P2 coins), small held-out val, micro-batch 8 + accumulation 4. Also save an embedding index of the training set for nearest-neighbour explanations.
2. **Gradio app** (`app/`): upload or webcam → coin crop (reuse the cropping code from `extract_frames_p2.py`) → denomination + confidence for all 5 classes → explanation: 5 nearest dataset images (DINOv2 embeddings) and a heatmap → inference time and device. Optionally show both the classifier answer and the kNN vote.
3. **Report** (6–8 pages): dataset update; DINOv2 explained in Neurohive style with a diagram and a link to the relevant Neurohive article (find and verify the link); baseline vs SOTA table; learning curves; ablation; explainability; ethics & limitations (currency images, no synthetic data, dataset bias, P2 videos recorded by another person); reproducibility.
4. **Demo video** script, README update, final repo check.

## 7. Paper — what is left (after Assignment 2)

Sections ready: Related Work (`related_work.md`). Outline: `PAPER_STRUCTURE.md`.
To write: Dataset, Models and Training, Experimental Protocol, Results (Exp. 1, Exp. 2), Error Analysis and Explainability, Discussion, Limitations, Conclusion, Abstract, Introduction. Then LaTeX (ELCVIA template on Overleaf, 8–12 pages, 11 pt, single-spaced), BibTeX, author info for both authors, AI-use disclosure statement, dataset release (GitHub and/or Zenodo), submission, confirmation email with manuscript number to the teacher.

Still needed from the authors: coin table (P2 coin, issue year, condition), phone models for both parts, light source for P2, exact P1 image count (`results_paper/p1_counts.csv`).

### Writing rules (requested by Sabira)
- Natural but formal academic English, no filler. Avoid: "delve", "crucial", "pivotal", "landscape", "comprehensive", "notably", "it is worth noting", stacked "Furthermore/Moreover", habitual lists of three, heavy dash use, unsupported "novel".
- Do **not** frame the work as "the problem of training on small datasets"; present it positively, focusing on what the study shows.
- Do **not** open with the 20 vs 50 tenge example; in the introduction just say some denominations share colour and design.
- Part 2 conditions are identical: same place, same background, same phone.
- Explain why testing and validation use held-out Part 2 coins in simple words: we want to know whether a model recognises coins it has never seen, so those coins are hidden from training; both models are tested on the same coins (same exam), choose their best epoch with the same validation coin (same judge), and use the same number of training images.
- Use "we". Never invent numbers or references; verify every citation; mark unknowns as [TODO].

### Verified references (numeric IEEE style)
Cirillo et al. 2024 (Neural Comput. Appl., modern coin classification); Kornblith et al. 2019 (CVPR, ImageNet transfer); Oquab et al. 2024 (TMLR, DINOv2); Barbu et al. 2019 (NeurIPS, ObjectNet); Geirhos et al. 2020 (Nat. Mach. Intell., shortcut learning); Barz & Denzler 2020 (J. Imaging, near-duplicates); Selvaraju et al. 2017 (ICCV, Grad-CAM). Full entries in `related_work.md`. Still to verify for the methods section: ResNet, ViT, ImageNet, label smoothing, AdamW.

## 8. Working conventions
- Ask before deleting results or rewriting finished work.
- Long trainings: keep them resumable and log progress to a file; put cheap models first.
- Test logic on small synthetic data before long runs when possible.
- Keep `assignment1/` untouched.

## 9. Assignment 2 progress (added 6 Nov 2026)
- `coinlib/` = shared code (data, DINOv2, resumable training, cropping, `predictor.py`). `paper_experiments.ipynb` is untouched and does not import it.
- `a2_final_training.ipynb` + `a2/RUN_FINAL_TRAINING.md`: final model (train P1 + c1–c6, val c7, test c8). NOT RUN yet (user decided to run it later). Its weights must never be used for paper numbers.
- `app/`: phone web app (FastAPI + canvas), approved mock-up in `app/mockup/`. Works end to end with the development model (fold-1 weights, index in `results_a2_dev/`); camera on a real phone, sound and the tap-to-select fallback are not yet tested on a device. See `app/README.md`.
- Still to do: run the final training, rebuild/switch to the final model, real-phone test over https, report (`a2/`), demo video script, README update.

### Update 6 Nov 2026 (evening)
- Final model trained (`results_a2/`, 13 epochs, 17.6 min): held-out coin c8 accuracy 0.834, macro F1 0.830 (kNN k=10 0.839); validation coin c7 about 0.73. Confusions on c8: 5->10 (20 of 40), 50->20 (7), 10->5 (5). The app uses it automatically.
- External check on 7 phone photos of 5 other coins on squared paper (`size_test/`): model alone 29/31, with size 31/31 (size means were measured on the same photos, so the second number is optimistic). The fold-1 development model got 9/31 there.
- Size from squared paper (`coinlib/gridscale.py`, 1 cell = 5 mm): diameters in cells 5: 3.59, 20: 3.80, 10: 3.96, 50: 4.70, 100: 5.25 (one specimen each). Used as a soft likelihood (SIGMA 0.15, floor 0.05), only when a grid is found (strength >= 0.3, x/y cell sizes agree within 6%). Distance to the sheet is an estimate from an assumed 69 degree field of view.
- Focus gate: a capture starts only when the coin-region sharpness has settled (app.js `settled`, `?sharpmin=` overrides the floor 80).
- Report A2 (Russian draft): `a2/report/Assignment2_report_ru.docx`, built by `a2/report/build_report.js` (figures in `a2/report/fig/`, made by `make_figures.py` and from `results_paper/figures/`). Verified 8 pages in Word before the last wording edits. Open TODOs inside: GitHub link, video link, co-author consent, where the weights file lives, AI-use statement. DINOv2 is cited from the original paper because no Neurohive article was found. Verified refs added: ViT, DINO, iBOT, ResNet, AdamW, label smoothing (see the report's reference list).
