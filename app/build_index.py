"""Embed training images with a trained model and save the nearest-neighbour index.

Development index (weights of fold 1 of the paper experiments: trained on P2 coins c3-c8):
    python app/build_index.py --weights results_paper/weights/E2__dinov2_s__P2__f1__s42.pt \
        --out results_a2_dev/index.npz --p2-coins c3,c4,c5,c6,c7,c8 --no-p1
The final model writes its own index from a2_final_training.ipynb.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coinlib.config import ROOT
from coinlib.data import load_part1, load_part2
from coinlib.models import DEVICE, DinoClassifier
from coinlib.train import predict
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--p2-coins", default=""); ap.add_argument("--no-p1", action="store_true")
a = ap.parse_args()
frames = [] if a.no_p1 else [load_part1()]
if a.p2_coins:
    p2 = load_part2(); frames.append(p2[p2.coin.isin(a.p2_coins.split(","))])
fr = pd.concat(frames, ignore_index=True)
model = DinoClassifier().to(DEVICE); model.load_state_dict(torch.load(a.weights, map_location="cpu")); model.eval()
print(f"embedding {len(fr)} images on {DEVICE}", flush=True)
_, emb = predict(model, fr)
root = str(ROOT) + "/"
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, emb=emb.astype("float32"), paths=np.array([p.replace(root, "") for p in fr.path]),
                    label=fr.label.to_numpy(), coin=fr.coin.to_numpy(), side=fr.side.to_numpy(), part=fr.part.to_numpy())
print("saved", a.out)
