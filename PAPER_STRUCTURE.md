# Paper structure

- Recognising Coins, Not Specimens: Kazakhstan Tenge Classification with Few Physical Samples


**Target journal:** ELCVIA – Electronic Letters on Computer Vision and Image Analysis (8–12 pages, single-spaced, 11 pt, figures and tables in the text).

**Status:**  can be written now ·  waits for experiment results

---

## Abstract 
The task, the two-part controlled design, the main numbers from both experiments and the answer to the research question in one sentence. Written last.

## 1. Introduction 
- Why tenge coins are a non-trivial case: 20 and 50 tenge share colour and design and differ mainly in diameter; 10, 20 and 50 tenge share the reverse emblem.
- Practical uses: coin counting, assistive apps for visually impaired users.
- The general problem: datasets built from a few physical objects may teach a model the objects rather than the class.
- Preliminary observation: one coin per denomination, ResNet18 at 0.969, equal accuracy on both sides, all errors on 20 tenge.
- Research question and the idea of varying one factor at a time.
- Contributions (without result numbers) and an outline of the paper.

## 2. Related Work 
- Coin recognition: hand-crafted features, then CNN-based approaches.
- Transfer learning on small datasets.
- Self-supervised vision transformers (ViT, DINOv2).
- Shortcut learning and data leakage in evaluation.
- Explainability with Grad-CAM.

Only verified references.

## 3. Dataset 
### 3.1 Collection pipeline
Smartphone videos instead of photos; sharpest-frame selection in 20 time windows; three-step coin localisation (Hough circle  ellipse fit on brightness and saturation edges  nearest frame); square crop with a random margin of 1.3–3.0 coin radii; manual check; splitting by video or coin rather than by frame.

### 3.2 Part 1: one specimen, varied conditions
One coin per denomination, six background  lighting setups (table), both sides, 60 videos. Notes on recording: camera tilt up to about 45°, changing distance, coin turned by hand.

### 3.3 Part 2: varied specimens, fixed conditions
Five coins per denomination (25 coins), white background and constant lighting, 50 videos, 987 images. Explicit justification for fixing the conditions. Table of specimens (issue year, condition) and image counts per coin.

## 4. Models and Training 
Simple CNN, ResNet18 and DINOv2 ViT-S/14; preprocessing; augmentation with a reason for each choice (full rotation, weak colour jitter, no flips); two-stage fine-tuning for pretrained models; hyperparameter table; best-epoch selection and early stopping on validation loss.

## 5. Experimental Protocol 
### 5.1 Experiment 1: new conditions, same coin
Train on setups 2, 4, 5, 6; validate on 3; test on 1 (no daylight in training). Three seeds.

### 5.2 Experiment 2: new coins, same conditions
Leave-one-specimen-out over 5 folds. P1-trained and P2-trained versions of each model evaluated on the same test coins, with the same validation coin and equal training-set sizes. Why the P1 training set includes the white background, and why this is not leakage.

### 5.3 Metrics
Accuracy and macro-F1 as mean  std; confusion matrices; per-coin and per-side accuracy; definition of the specimen gap.

## 6. Results: Generalisation Across Conditions 
Experiment 1 table for the three models (mean  std over seeds) and training curves.

## 7. Results: Generalisation Across Specimens 
P1-trained vs P2-trained accuracy for each model; specimen gap; per-fold and per-coin accuracy; accuracy by coin side; confusion matrices.

## 8. Error Analysis and Explainability 
Which denominations and specimens are confused and why (20 vs 50 tenge, loss of absolute size); Grad-CAM of the P1- and P2-trained ResNet18 on the same unseen coins; DINOv2 nearest neighbours showing which training coins a prediction relies on.

## 9. Discussion 
The answer to the research question; how the three types of pretraining differ in specimen generalisation; practical advice for collecting small object datasets: what to vary first, conditions or objects.

## 10. Limitations 
Single background in Part 2 (by design); different days, people and phones for the two parts; regular issues only; correlated video frames; loss of absolute coin size after cropping; Part 2 validation coin also used for epoch selection of the P1-trained models; only five specimens per denomination.

## 11. Conclusion 
Main findings in a few sentences and next steps: more specimens and issue years, varied conditions for many specimens, keeping a size reference.

## Data and Code Availability 
Dataset, extraction scripts and experiment notebook in a public GitHub repository.

## References
Verified sources only.

---
