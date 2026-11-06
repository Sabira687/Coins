"""Resumable two-stage training for the DINOv2 classifier.

Same recipe as the paper experiments (stage 1: head only; stage 2: whole network, AdamW, cosine decay, label
smoothing 0.05, micro-batches of 8 with gradient accumulation over 4). The state is saved after every epoch, so
an interrupted run continues from the last finished epoch.
"""
import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import *
from .data import make_loaders, CoinDS, eval_tf
from .models import DEVICE, DinoClassifier

CFG = dict(head_epochs=5, ft_epochs=15, ft_lr=1e-5, patience=5, batch=8, accum=4)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _cpu(sd):
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def run_epoch(model, loader, criterion, optimizer=None, accum=1):
    training = optimizer is not None
    model.train(training)
    tot, correct, n = 0.0, 0, 0
    if training: optimizer.zero_grad()
    with torch.set_grad_enabled(training):
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x); loss = criterion(out, y)
            if training:
                (loss / accum).backward()
                if (i + 1) % accum == 0 or i + 1 == len(loader):
                    optimizer.step(); optimizer.zero_grad()
            tot += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot / n, correct / n


def _stage_optimizer(model, stage, cfg):
    if stage == 0:
        for p in model.parameters(): p.requires_grad = False
        for p in model.head.parameters(): p.requires_grad = True
        return torch.optim.Adam(model.head.parameters(), lr=1e-3), None
    for p in model.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg["ft_lr"]},
                             {"params": model.head.parameters(), "lr": cfg["ft_lr"] * 10}], weight_decay=0.05)
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["ft_epochs"])


def train_dino(tr, va, seed, ckpt_path, log=print, cfg=None, balanced=True):
    """Train DINOv2 ViT-S/14 on frame `tr`, choose the epoch by validation loss on `va`.
    Returns (model with best weights, history dict)."""
    cfg = {**CFG, **(cfg or {})}
    ckpt_path = Path(ckpt_path); ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model = DinoClassifier().to(DEVICE)
    trl, val, _ = make_loaders(tr, va, va, seed, cfg["batch"], balanced=balanced)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    stage_epochs = [cfg["head_epochs"], cfg["ft_epochs"]]
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, bad = float("inf"), _cpu(model.state_dict()), 0
    stage, ep, finished = 0, 0, False
    opt_state = sched_state = None

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        hist, best_loss, best_state, bad = ck["hist"], ck["best_loss"], ck["best_state"], ck["bad"]
        stage, ep, finished = ck["stage"], ck["ep"], ck["finished"]
        opt_state, sched_state = ck["opt"], ck["sched"]
        log(f"  resumed from checkpoint: stage {stage + 1}, {len(hist['val_loss'])} epochs done")

    while not finished:
        opt, sched = _stage_optimizer(model, stage, cfg)
        if opt_state is not None:
            opt.load_state_dict(opt_state)
            if sched is not None and sched_state is not None: sched.load_state_dict(sched_state)
            opt_state = sched_state = None
        patience = cfg["head_epochs"] if stage == 0 else cfg["patience"]
        while ep < stage_epochs[stage] and bad < patience:
            t0 = time.time()
            tl, ta = run_epoch(model, trl, criterion, opt, cfg["accum"])
            vl, vacc = run_epoch(model, val, criterion)
            if sched is not None: sched.step()
            for k, v in zip(hist, (tl, vl, ta, vacc)): hist[k].append(v)
            ep += 1
            if vl < best_loss:
                best_loss, best_state, bad = vl, _cpu(model.state_dict()), 0
            else:
                bad += 1
            log(f"    epoch {len(hist['val_loss']):>2} | train loss {tl:.3f} acc {ta:.3f} | "
                f"val loss {vl:.3f} acc {vacc:.3f} | {time.time() - t0:.0f}s")
            last = ep >= stage_epochs[stage] or bad >= patience
            nxt_stage, nxt_ep, nxt_bad = (stage, ep, bad)
            done = False
            if last:                                    # end of stage: continue from the best weights
                model.load_state_dict(best_state)
                if stage == 0:
                    nxt_stage, nxt_ep, nxt_bad = 1, 0, 0
                    done = cfg["ft_epochs"] == 0
                else:
                    done = True
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(dict(model=_cpu(model.state_dict()), hist=hist, best_loss=best_loss, best_state=best_state,
                            bad=nxt_bad, stage=nxt_stage, ep=nxt_ep, finished=done,
                            opt=None if last else opt.state_dict(),
                            sched=None if (last or sched is None) else sched.state_dict()), tmp)
            os.replace(tmp, ckpt_path)
            if last:
                bad = nxt_bad
                break
        if stage == 1 or (stage == 0 and cfg["ft_epochs"] == 0):
            finished = True
        else:
            stage, ep, bad = 1, 0, 0
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def predict(model, frame, batch=16):
    """Probabilities, and L2-normalised 384-d embeddings (cosine similarity = dot product), for every row of `frame` (no augmentation)."""
    model.eval()
    probs, embs = [], []
    loader = torch.utils.data.DataLoader(CoinDS(frame, eval_tf), batch)
    for x, _ in loader:
        e = model.embed(x.to(DEVICE))
        probs.append(F.softmax(model.head(e), 1).cpu()); embs.append(F.normalize(e, dim=1).cpu())
    return torch.cat(probs).numpy(), torch.cat(embs).numpy()
