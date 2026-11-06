import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

from .config import *

P1_RE = re.compile(r"_([AR])(\d+)_\d+\.jpg$", re.IGNORECASE)
P2_RE = re.compile(r"_c(\d+)([AR])_\d+\.jpg$", re.IGNORECASE)
side_of = lambda s: "obverse" if s.upper() == "A" else "reverse"


def load_part1(p1_dir=P1_DIR):
    rows = []
    for split in ["train", "val", "test"]:
        for c in CLASSES:
            for p in sorted((p1_dir / split / c).glob("*.jpg")):
                m = P1_RE.search(p.name)
                if m:
                    rows.append(dict(path=str(p), cls=c, label=C2I[c], coin="c0",
                                     side=side_of(m.group(1)), setup=int(m.group(2)), part="P1"))
    return pd.DataFrame(rows)


def load_part2(p2_dir=P2_DIR):
    rows = []
    for c in CLASSES:
        for p in sorted((p2_dir / c).glob("*.jpg")):
            m = P2_RE.search(p.name)
            if m:
                rows.append(dict(path=str(p), cls=c, label=C2I[c], coin=f"c{int(m.group(1))}",
                                 side=side_of(m.group(2)), setup=0, part="P2"))
    return pd.DataFrame(rows)


# Same preprocessing and augmentation as Assignment 1 and the paper experiments
train_tf = transforms.Compose([
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.25, p=1.0)], p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.02),
    transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class CoinDS(Dataset):
    def __init__(self, frame, tf):
        self.paths, self.labels, self.tf = frame.path.tolist(), frame.label.tolist(), tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def class_balanced_sampler(frame, seed):
    """Draw len(frame) samples per epoch with equal weight per (class, source): P1 has ~1100 images, P2 ~1200,
    so without this the image counts per class and part differ slightly."""
    key = frame.label.astype(str) + "_" + frame.part
    w = 1.0 / key.map(key.value_counts()).to_numpy()
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(frame), replacement=True, generator=g)


def make_loaders(tr, va, te, seed, batch, balanced=False):
    g = torch.Generator().manual_seed(seed)
    if balanced:
        trl = DataLoader(CoinDS(tr, train_tf), batch, sampler=class_balanced_sampler(tr, seed))
    else:
        trl = DataLoader(CoinDS(tr, train_tf), batch, shuffle=True, generator=g)
    return trl, DataLoader(CoinDS(va, eval_tf), batch), DataLoader(CoinDS(te, eval_tf), batch)


def denorm(t):
    return np.clip(t.permute(1, 2, 0).cpu().numpy() * np.array(STD) + np.array(MEAN), 0, 1)
