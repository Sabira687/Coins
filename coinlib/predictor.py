"""Inference wrapper used by the app: coin detection, classification, nearest neighbours, attention map."""
import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import *
from .cropping import locate_coin, crop_coin
from .data import eval_tf
from .gridscale import measure, combine
from .models import DEVICE, DinoClassifier


def _sync():
    if DEVICE == "mps": torch.mps.synchronize()
    elif DEVICE == "cuda": torch.cuda.synchronize()


def _jpeg_b64(rgb, quality=82):
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class CoinPredictor:
    def __init__(self, weights, index_path, uncertain_below=0.55, margin=2.0, k=5):
        self.uncertain_below, self.margin, self.k = uncertain_below, margin, k
        self.model = DinoClassifier().to(DEVICE)
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()
        self._qkv = None
        self.model.backbone.blocks[-1].attn.qkv.register_forward_hook(lambda m, i, o: setattr(self, "_qkv", o.detach()))
        self.heads = self.model.backbone.blocks[-1].attn.num_heads
        d = np.load(index_path, allow_pickle=True)   # own index file (object arrays for coin/side)
        self.emb = d["emb"].astype("float32")
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True) + 1e-9
        self.paths, self.label, self.coin, self.side = d["paths"], d["label"], d["coin"], d["side"]
        self._init_map()
        self.classify(np.full((300, 300, 3), 128, np.uint8), (0.5, 0.5, 0.3))     # warm-up

    # ---------- embedding map: supervised 2D projection (LDA) of the reference embeddings, fixed for the whole index
    def _init_map(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.lda = LinearDiscriminantAnalysis(n_components=2).fit(self.emb, self.label)
        z = self.lda.transform(self.emb)
        self.lo, self.hi = np.percentile(z, 1, 0), np.percentile(z, 99, 0)
        rng = np.random.default_rng(0)
        pick = np.concatenate([rng.permutation(np.where(self.label == c)[0])[:40] for c in range(len(CLASSES))])
        self.map_points = [dict(x=float(v[0]), y=float(v[1]), c=int(self.label[i]))
                           for i, v in zip(pick, self._norm2(z[pick]))]

    def _norm2(self, z):
        return np.clip((z - self.lo) / (self.hi - self.lo), -.05, 1.05)

    # ---------- detection
    def detect(self, bgr):
        c, method = locate_coin(bgr)
        if c is None: return None
        h, w = bgr.shape[:2]; m = min(h, w)
        cx, cy, r = c
        if not (0.05 * m <= r <= 0.48 * m): return None
        return dict(cx=float(cx / w), cy=float(cy / h), r=float(1.06 * r / m), method=method, sharp=self.sharpness(bgr, (cx, cy, 1.06 * r)))

    @staticmethod
    def sharpness(bgr, circle):
        """Focus measure of the coin region: variance of the Laplacian of the coin square resized to 96x96.
        Sharp 480 px frames give about 250-600, a Gaussian blur of sigma 1 px drops it to 30-50 (tested on sample frames)."""
        h, w = bgr.shape[:2]; cx, cy, r = circle
        x0, x1, y0, y1 = int(max(0, cx - r)), int(min(w, cx + r)), int(max(0, cy - r)), int(min(h, cy + r))
        if x1 - x0 < 8 or y1 - y0 < 8: return 0.0
        roi = cv2.resize(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
        return float(cv2.Laplacian(roi, cv2.CV_64F).var())   # 6% wider: the fitted circle sits slightly inside the rim

    # ---------- classification
    @torch.no_grad()
    def classify(self, bgr, circle, margin=None, use_size=True):
        t_all = time.perf_counter()
        h, w = bgr.shape[:2]; m = min(h, w)
        cx, cy, r = circle
        crop = crop_coin(bgr, (cx * w, cy * h, r * m), margin or self.margin, 512)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = eval_tf(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
        _sync(); t0 = time.perf_counter()
        e = self.model.embed(x); logits = self.model.head(e)
        _sync(); model_ms = (time.perf_counter() - t0) * 1000
        prob = F.softmax(logits, 1)[0].cpu().numpy()
        q = F.normalize(e, dim=1)[0].cpu().numpy()
        # attention of the CLS token (last block, mean over heads) on the 16x16 patch grid
        qkv = self._qkv.reshape(1, -1, 3, self.heads, self._qkv.shape[-1] // 3 // self.heads)
        qh, kh = qkv[:, :, 0].permute(0, 2, 1, 3), qkv[:, :, 1].permute(0, 2, 1, 3)
        att = torch.softmax(qh[:, :, :1] @ kh.transpose(-1, -2) / qh.shape[-1] ** .5, -1)[0, :, 0, 1:].mean(0)
        g = int(att.numel() ** .5); att = att.reshape(g, g).cpu().numpy()
        # single high-norm background tokens are a known ViT artefact: a 3x3 median removes isolated peaks
        att = cv2.medianBlur(att.astype(np.float32), 3)
        att = np.clip((att - att.min()) / (np.percentile(att, 99) - att.min() + 1e-9), 0, 1)
        # neighbours and kNN vote
        sims = self.emb @ q
        top = np.argsort(-sims)[:self.k]
        votes = np.zeros(len(CLASSES))
        for i in top: votes[self.label[i]] += sims[i]
        knn = int(votes.argmax())
        qm = self._norm2(self.lda.transform(q[None]))[0]
        size = measure(bgr, circle) if use_size else None
        prob_model = prob
        if size and size.get("ok"): prob = combine(prob, size["cells"])
        pred = int(prob.argmax()); conf = float(prob[pred]); order = np.argsort(-prob)
        return dict(
            probs=[float(p) for p in prob], probs_model=[float(p) for p in prob_model], size=size, pred=pred, label=SHORT[pred], conf=conf,
            uncertain=conf < self.uncertain_below, second=int(order[1]), second_conf=float(prob[order[1]]),
            neighbours=[dict(path=str(self.paths[i]), sim=float(sims[i]), cls=int(self.label[i]),
                             coin=str(self.coin[i]), side=str(self.side[i])) for i in top],
            knn=dict(pred=knn, count=int((self.label[top] == knn).sum()), k=self.k),
            attention=[[round(float(v), 3) for v in row] for row in att],
            crop=_jpeg_b64(cv2.cvtColor(cv2.resize(crop, (288, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)),
            map=dict(points=self.map_points, query=dict(x=float(qm[0]), y=float(qm[1]))),
            timing=dict(model_ms=model_ms, total_ms=(time.perf_counter() - t_all) * 1000),
            device=DEVICE)
