# Dataset and model weights

Images and weights are not stored in this repository (they would add about 450 MB). They are hosted on Google Drive:

**Google Drive folder: [TODO: link]**

| File | Content | Size |
|---|---|---|
| `dataset.zip` | Part 1: `dataset/<train\|val\|test>/<class>/<class>_<A\|R><setup>_<k>.jpg` (1109 images, one coin, 6 setups) | ~50 MB |
| `dataset_p2.zip` | Part 2: `dataset_p2/<class>/<class>_c<coin><A\|R>_<k>.jpg` (1585 images, 40 coins) | ~57 MB |
| `weights_paper.zip` | fold-1 weights of ResNet18 and DINOv2 (P1- and P2-trained) used for the Grad-CAM and neighbour figures | ~250 MB |
| `final_dinov2.pt` | final Assignment 2 model used by the app (trained on 31 of 41 coins; not for paper numbers) | ~85 MB |
| `resnet18_coins.pt` | Assignment 1 ResNet18 | ~44 MB |

## How to install

Unzip into the repository root so that the folders `dataset/` and `dataset_p2/` sit next to `paper_experiments.ipynb`; put `final_dinov2.pt` into `results_a2/`, `resnet18_coins.pt` into `assignment1/results/`, and the contents of `weights_paper.zip` into `results_paper/weights/`.

Names carry the metadata: `A` is the obverse, `R` the reverse, `c0` the Part 1 coin, `c1`–`c8` the Part 2 coins. Raw videos are not published.

Licence and citation: [TODO: choose licence, e.g. CC BY 4.0; add Zenodo DOI if a permanent copy is made].
