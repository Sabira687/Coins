"""Scale from squared paper: cell size in pixels near a coin, coin diameter in cells (1 cell = 5 mm)."""
import cv2
import numpy as np

CELL_MM = 5.0


def coin_masks(bgr, min_area_frac=0.0008):
    """Segment coins on squared paper. Returns list of dicts (mask, centre, equivalent diameters in px)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    med = cv2.medianBlur(gray, 9)                       # thin grid lines disappear
    paper = np.median(med[med > np.percentile(med, 60)])
    sat = cv2.medianBlur(hsv[..., 1], 9)
    gold = (sat > 38) & (hsv[..., 0] > 8) & (hsv[..., 0] < 45)
    m = ((med < paper - 28) | gold).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area_frac * h * w or len(c) < 20: continue
        (ex, ey), (ma, mi), ang = cv2.fitEllipse(c)
        if max(ma, mi) / min(ma, mi) > 1.25 or a < 0.9 * np.pi * ma * mi / 4: continue     # not a coin (handwriting, shadows)
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2: continue
        mk = np.zeros((h, w), np.uint8); cv2.drawContours(mk, [c], -1, 255, -1)
        out.append(dict(contour=c, mask=mk, cx=ex, cy=ey, d_area=float(np.sqrt(4 * a / np.pi)), d_ellipse=float(np.sqrt(ma * mi)),
                        d_major=float(max(ma, mi)), d_minor=float(min(ma, mi))))
    return out


def _profile(d, axis, band, exclude):
    """1-D profile of line darkness: median over a band of rows (axis=1) or columns (axis=0), ignoring excluded pixels."""
    x = d.astype(np.float32).copy(); x[exclude > 0] = np.nan
    sub = x[band[0]:band[1], :] if axis == 1 else x[:, band[0]:band[1]].T
    with np.errstate(all="ignore"):
        p = np.nanmedian(sub, axis=0)
    return np.nan_to_num(p - np.nanmean(p))


def _lattice_period(p, pmin=12, pmax=90):
    p = p - np.convolve(p, np.ones(61) / 61, mode="same")
    ac = np.correlate(p, p, "full")[len(p) - 1:]; ac /= ac[0] + 1e-9
    lag = pmin + int(np.argmax(ac[pmin:pmax])); strength = float(ac[lag])
    # refine: positions of the lines on the lattice, then a linear fit position = x0 + k * period
    s = np.convolve(p, np.ones(3) / 3, mode="same"); period = float(lag)
    for _ in range(3):
        ph = np.angle(np.sum(s * np.exp(-2j * np.pi * np.arange(len(s)) / period)))
        x0 = (-ph / (2 * np.pi) * period) % period
        ks, xs = [], []
        for k in range(int((len(s) - x0) / period)):
            c = x0 + k * period; lo, hi = int(c - period / 4), int(c + period / 4) + 1
            if lo < 1 or hi >= len(s) - 1: continue
            seg = s[lo:hi]; j = int(np.argmax(seg)); i = lo + j
            if seg[j] < 0.25 * np.max(s): continue
            a, b, cc = s[i - 1], s[i], s[i + 1]; den = a - 2 * b + cc
            xs.append(i + (0.5 * (a - cc) / den if abs(den) > 1e-9 else 0)); ks.append(k)
        if len(ks) < 6: break
        period = float(np.polyfit(ks, xs, 1)[0])
    return period, len(ks), strength


def cell_size(bgr, coin, all_coins, margin=1.0):
    """Cell size in px next to one coin: line spacing measured in a band of rows and a band of columns through the
    coin, with all coins excluded. Returns (cell_px, details)."""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = np.clip(cv2.GaussianBlur(gray, (0, 0), 12) - cv2.GaussianBlur(gray, (0, 0), 0.8), 0, None)
    excl = np.zeros((h, w), np.uint8)
    for c in all_coins: excl = np.maximum(excl, cv2.dilate(c["mask"], np.ones((15, 15), np.uint8)))
    r = coin["d_area"] / 2
    rows = (int(max(0, coin["cy"] - margin * r)), int(min(h, coin["cy"] + margin * r)))
    cols = (int(max(0, coin["cx"] - margin * r)), int(min(w, coin["cx"] + margin * r)))
    px, nx, sx = _lattice_period(_profile(d, 1, rows, excl))      # spacing of vertical lines along x
    py, ny, sy = _lattice_period(_profile(d, 0, cols, excl))      # spacing of horizontal lines along y
    return (px + py) / 2, dict(px=px, py=py, nx=nx, ny=ny, strength=min(sx, sy))


# ---------- size as evidence for the denomination
# Mean diameter in cells of one coin per denomination, measured on 7 photos (size_test/diameters.csv, readings where the cell
# size from x and y agreed within 6%). One specimen per denomination, so SIGMA is deliberately wider than the measured spread.
EXPECTED_CELLS = {"5": 3.59, "10": 3.96, "20": 3.80, "50": 4.70, "100": 5.25}
SIGMA, EPS = 0.15, 0.05
FOV_LONG_SIDE_DEG = 69.0          # assumed horizontal field of view of the long image side of a phone main camera (about 26 mm equivalent)


def measure(bgr, circle, max_side=1280):
    """Coin diameter in cells next to the detected coin. circle = (cx, cy, r) as fractions of the frame.
    Returns None if no coin was found, else a dict with ok=False (no usable grid) or ok=True and the size."""
    h, w = bgr.shape[:2]; k = min(1.0, max_side / max(h, w))
    im = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else bgr
    cx, cy, r = circle[0] * w * k, circle[1] * h * k, circle[2] * min(h, w) * k
    coins = coin_masks(im)
    best = min(coins, key=lambda c: np.hypot(c["cx"] - cx, c["cy"] - cy), default=None)
    if best is None or np.hypot(best["cx"] - cx, best["cy"] - cy) > 0.5 * r or not 0.6 < best["d_area"] / (2 * r) < 1.5:
        return None
    cell, det = cell_size(im, best, coins)
    dis = abs(det["px"] - det["py"]) / cell
    if min(det["nx"], det["ny"]) < 5 or det["strength"] < 0.3 or dis > 0.06:   # strength: plain paper gives <= 0.11, grids 0.45-0.68
        return dict(ok=False, reason="no clear squared paper", disagree=float(dis))
    cells = best["d_area"] / cell
    f_px = (max(h, w) * k / 2) / np.tan(np.radians(FOV_LONG_SIDE_DEG / 2))
    return dict(ok=True, cells=float(cells), mm=float(cells * CELL_MM), cell_px=float(cell / k), disagree=float(dis),
                distance_cm=float(f_px * (CELL_MM / 10) / cell))


def combine(probs, cells):
    """Multiply the model probabilities by the likelihood of the measured size (with a floor, so one bad measurement cannot veto)."""
    mu = np.array([EXPECTED_CELLS[s] for s in ["5", "10", "20", "50", "100"]])
    lik = EPS + (1 - EPS) * np.exp(-0.5 * ((cells - mu) / SIGMA) ** 2)
    post = np.asarray(probs) * lik
    return post / post.sum()
