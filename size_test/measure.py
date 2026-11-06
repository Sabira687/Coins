"""Coin diameter in notebook cells (1 cell = 5 mm) for the five-coin sheets in size_test/.
Layout of every sheet: left column top to bottom 50, 20, 100 tenge; right column top to bottom 10, 5 tenge.
    python size_test/measure.py   ->  size_test/diameters.csv
"""
import sys, glob, csv
from pathlib import Path
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coinlib.gridscale import coin_masks, cell_size

rows = []
for f in sorted(glob.glob(str(Path(__file__).parent / "sheet_*.jpg"))):
    im = cv2.imread(f); h, w = im.shape[:2]; cs = coin_masks(im)
    left = sorted([c for c in cs if c["cx"] < w / 2], key=lambda c: c["cy"]); right = sorted([c for c in cs if c["cx"] >= w / 2], key=lambda c: c["cy"])
    named = (list(zip(["50", "20", "100"], left)) if len(left) == 3 else [("50", left[0])] if len(left) == 1 else []) + (list(zip(["10", "5"], right)) if len(right) == 2 else [])
    for name, c in named:
        cell, det = cell_size(im, c, cs)
        rows.append(dict(sheet=Path(f).name, coin=name, diameter_px=round(c["d_area"], 1), cell_px=round(cell, 2),
                         cells=round(c["d_area"] / cell, 3), mm=round(c["d_area"] / cell * 5, 2),
                         xy_disagree_pct=round(abs(det["px"] - det["py"]) / cell * 100, 1)))
with open(Path(__file__).parent / "diameters.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
import pandas as pd
d = pd.DataFrame(rows); good = d[d.xy_disagree_pct <= 6]
print(good.groupby("coin").cells.agg(["mean", "std", "count"]).round(3).reindex(["5", "20", "10", "50", "100"]))
