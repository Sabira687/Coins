"""Report figures that are not produced by the notebooks: DINOv2 scheme, size on squared paper, app screenshots strips.
    python a2/report/make_figures.py
(the screenshot strips need the PNG screenshots in a2/report/shots/, taken from the running app)"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).parent; ROOT = HERE.parent.parent

def box(ax, x, y, w, h, txt, fc="#eef3fb", ec="#2f5d9b", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.12", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", color="#111", linespacing=1.3)
def arrow(ax, x0, y0, x1, y1, c="#444", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10, lw=1.2, color=c, connectionstyle=f"arc3,rad={rad}", linestyle=ls))

fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 25); ax.set_ylim(0, 9); ax.axis("off")
ax.text(0.2, 8.5, "A. DINOv2 pre-training (Meta AI, no labels)", fontsize=10, weight="bold", color="#2f5d9b")
box(ax, 0.2, 3.8, 2.1, 1.6, "image\n(LVD-142M)", fc="#f4f4f4", ec="#777")
box(ax, 3.3, 5.4, 3.0, 1.5, "global crops\n+ masked patches", fc="#f4f4f4", ec="#777")
box(ax, 3.3, 2.3, 3.0, 1.5, "local crops", fc="#f4f4f4", ec="#777")
arrow(ax, 2.35, 4.9, 3.25, 6.0); arrow(ax, 2.35, 4.3, 3.25, 3.1)
box(ax, 7.4, 5.4, 3.2, 1.5, "Teacher ViT\n(EMA of student)", fc="#fff1dc", ec="#c47a10", bold=True)
box(ax, 7.4, 2.3, 3.2, 1.5, "Student ViT", fc="#e4f3e7", ec="#2e8b4a", bold=True)
arrow(ax, 6.35, 6.15, 7.35, 6.15); arrow(ax, 6.35, 3.05, 7.35, 3.05); arrow(ax, 6.35, 5.6, 7.35, 3.6)
box(ax, 11.5, 3.6, 3.5, 2.1, "DINO loss (image)\n+ iBOT loss (patches)\n+ KoLeo regulariser", fc="#fde7e7", ec="#b33")
arrow(ax, 10.65, 6.1, 11.45, 5.2); arrow(ax, 10.65, 3.1, 11.45, 4.2)
ax.text(8.9, 1.2, r"$\theta_t \leftarrow \lambda\,\theta_t + (1-\lambda)\,\theta_s$", ha="center", fontsize=10)
ax.text(12.2, 2.4, "gradients flow\nonly to the student", fontsize=7.5, color="#b33")
arrow(ax, 12.6, 3.55, 10.8, 2.05, c="#b33", rad=-0.2, ls="--")
ax.plot([15.6, 15.6], [0.3, 8.3], color="#bbb", lw=1, ls=":")
ax.text(16.0, 8.5, "B. Our coin classifier (fine-tuning)", fontsize=10, weight="bold", color="#2f5d9b")
box(ax, 16.0, 4.0, 2.0, 1.7, "coin crop\n224×224", fc="#f4f4f4", ec="#777")
gx, gy = 18.8, 3.95
ax.add_patch(Rectangle((gx, gy), 1.8, 1.8, fc="#fafafa", ec="#555", lw=1))
for i in range(1, 8):
    ax.plot([gx + i * 1.8 / 8] * 2, [gy, gy + 1.8], color="#999", lw=.5); ax.plot([gx, gx + 1.8], [gy + i * 1.8 / 8] * 2, color="#999", lw=.5)
ax.text(gx + 0.9, gy - 0.2, "16×16 patches\n(14×14 px each)", ha="center", va="top", fontsize=7)
arrow(ax, 18.05, 4.85, 18.75, 4.85)
box(ax, 21.4, 3.5, 3.4, 2.7, "ViT-S/14\n12 transformer\nblocks, CLS +\n256 patch tokens", fc="#e4f3e7", ec="#2e8b4a", bold=True)
arrow(ax, 20.65, 4.85, 21.35, 4.85)
box(ax, 16.0, 0.4, 3.1, 1.3, "CLS embedding\n384-d (L2-normalised)", fc="#eef3fb", ec="#2f5d9b", fs=7)
box(ax, 19.7, 0.4, 2.3, 1.3, "linear layer\n384 → 5", fc="#fff1dc", ec="#c47a10")
box(ax, 22.5, 0.4, 2.4, 1.3, "softmax:\n5 probabilities", fc="#fde7e7", ec="#b33")
arrow(ax, 22.6, 3.45, 18.0, 1.75, rad=0.1); arrow(ax, 19.15, 1.05, 19.65, 1.05); arrow(ax, 22.05, 1.05, 22.45, 1.05)
box(ax, 16.0, 6.9, 3.9, 1.2, "nearest images (cosine)\nand kNN vote", fc="#f4f4f4", ec="#777", fs=7)
box(ax, 20.6, 6.9, 4.2, 1.2, "CLS attention (last block)\n→ heat map", fc="#f4f4f4", ec="#777", fs=7)
arrow(ax, 17.2, 1.75, 17.5, 6.85, c="#888", ls="--"); arrow(ax, 23.1, 6.25, 22.9, 6.85, c="#888", ls="--")
plt.savefig(HERE / "fig/dinov2_scheme.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close()

d = pd.read_csv(ROOT / "size_test/diameters.csv"); g = d[d.xy_disagree_pct <= 6]
order = ["5", "20", "10", "50", "100"]; col = {"5": "#4aa3c7", "20": "#1faa59", "10": "#d9a21b", "50": "#d9627f", "100": "#8c6bd0"}
fig, ax = plt.subplots(figsize=(5.6, 2.5))
for i, c in enumerate(order):
    v = g[g.coin.astype(str) == c].cells.values; ax.scatter([i] * len(v) + np.random.default_rng(i).normal(0, .04, len(v)), v, s=22, color=col[c], zorder=3)
    ax.hlines(v.mean(), i - .28, i + .28, color="#222", lw=1.5, zorder=4)
ax.set_xticks(range(5)); ax.set_xticklabels([f"{c} tenge" for c in order]); ax.set_ylabel("diameter, cells (1 cell = 5 mm)")
ax.grid(axis="y", alpha=.3); ax.set_title("Coin diameter measured on squared paper (7 photos)", fontsize=9)
plt.tight_layout(); plt.savefig(HERE / "fig/size_cells.png", dpi=200, facecolor="white"); plt.close()
print("figures ok")
