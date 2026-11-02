"""
Part 2 (paper): 5 physical coins per denomination, one fixed setup
(white background, constant lighting). Only the specimen changes.

Input layout:
    raw_videos_p2/<class_name>/<coin><side>.mp4      e.g. raw_videos_p2/20_tenge/3R.mp4
        coin = 1..5   (physical coin ID inside the denomination)
        side = A (obverse, numeral) or R (reverse, emblem)

Output layout (no split folders: splits by coin are made in the notebook):
    dataset_p2/<class_name>/<class>_c<coin><side>_<k>.jpg
    dataset_p2/metadata.csv   (path, class, coin, video, side, sharpness, crop_method, margin)

Frame selection and coin cropping are identical to Part 1.

Usage:
    python extract_frames_p2.py            # raw_videos_p2 -> dataset_p2
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi"}
NAME_RE = re.compile(r"^(\d+)([AR])$", re.IGNORECASE)


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def thumb(gray):
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)


def resize_max(img, max_side):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def find_coin(frame, work_side=640):
    """Return (cx, cy, r) of the most likely coin in full-frame pixels, or None."""
    small = resize_max(frame, work_side)
    k = frame.shape[1] / small.shape[1]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    m = min(gray.shape)
    rmin, rmax = int(0.03 * m), int(0.48 * m)
    circles = None
    if hasattr(cv2, "HOUGH_GRADIENT_ALT"):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=m,
                                   param1=300, param2=0.8, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=m,
                                   param1=100, param2=45, minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return cx * k, cy * k, r * k


def _candidate_masks(small):
    """Binary images whose outer contours may outline the coin."""
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    k5 = np.ones((5, 5), np.uint8)

    # a) edges from brightness + colour saturation (metal is grey, skin/wood are not)
    g = cv2.GaussianBlur(clahe.apply(gray), (5, 5), 0)
    sb = cv2.GaussianBlur(sat, (5, 5), 0)
    edges = cv2.Canny(g, 30, 90) | cv2.Canny(sb, 15, 45)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k5, iterations=2)
    filled = edges.copy()
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cnts, -1, 255, thickness=cv2.FILLED)
    # opening with a big round kernel erases thin lines (palm lines, wood grain)
    # that touch the coin, but keeps the coin itself
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    yield cv2.morphologyEx(filled, cv2.MORPH_OPEN, k15)

    # b) low-saturation blob: grey/yellowish metal on a warmer, more colourful background
    med_sat = float(np.median(sb))
    if med_sat > 25:
        low_sat = (sb < 0.6 * med_sat).astype(np.uint8) * 255
        k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_OPEN, k9)
        low_sat = cv2.morphologyEx(low_sat, cv2.MORPH_CLOSE, k9, iterations=2)
        yield low_sat


def find_coin_ellipse(frame, work_side=640):
    """Fallback for tilted coins (ellipses) and busy backgrounds like a palm.
    Returns (cx, cy, r) with r = half of the ellipse's long axis, or None."""
    small = resize_max(frame, work_side)
    k = frame.shape[1] / small.shape[1]
    h, w = small.shape[:2]
    img_area = h * w

    best, best_score = None, 0.0
    for mask in _candidate_masks(small):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            if len(c) < 20:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2:
                continue  # touches the border: usually the hand / background, not the coin
            area = cv2.contourArea(c)
            if not (0.002 * img_area < area < 0.35 * img_area):
                continue
            (ex, ey), (a1, a2), _ = cv2.fitEllipse(c)
            major, minor = max(a1, a2), min(a1, a2)
            if major == 0 or minor / major < 0.35:
                continue  # too flat to be a coin
            fill = area / (np.pi * major * minor / 4)
            if not (0.85 < fill < 1.15):
                continue  # outline is not elliptical
            score = (1 - abs(1 - fill)) * np.sqrt(area)
            if score > best_score:
                best, best_score = (ex * k, ey * k, major / 2 * k), score
    return best


def locate_coin(frame):
    c = find_coin(frame)
    if c is not None:
        return c, "circle"
    c = find_coin_ellipse(frame)
    if c is not None:
        return c, "ellipse"
    return None, "none"


def crop_coin(frame, circle, margin, out_size):
    """Square crop centred on the coin; half-size = radius * margin.
    The square is shifted to stay inside the frame (no artificial padding)."""
    cx, cy, r = circle
    h, w = frame.shape[:2]
    side = min(int(2 * r * margin), h, w)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    crop = frame[y0:y0 + side, x0:x0 + side]
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def best_frames(path, n_windows):
    """Return the sharpest frame from each of n_windows equal time windows."""
    total = int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = count_frames(path)
    if total == 0:
        return []

    best = {}  # window -> (score, frame)
    cap = cv2.VideoCapture(str(path))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        win = min(idx * n_windows // total, n_windows - 1)
        gray = cv2.cvtColor(resize_max(frame, 400), cv2.COLOR_BGR2GRAY)
        score = sharpness(gray)
        if win not in best or score > best[win][0]:
            best[win] = (score, frame)
        idx += 1
    cap.release()
    return [best[w] for w in sorted(best)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_videos_p2")
    ap.add_argument("--dst", default="dataset_p2")
    ap.add_argument("--frames", type=int, default=20, help="frames per video")
    ap.add_argument("--max-side", type=int, default=800)
    ap.add_argument("--blur-ratio", type=float, default=0.35,
                    help="drop frames sharper than less than this * median sharpness of the video")
    ap.add_argument("--dup-thresh", type=float, default=0.0,
                    help="drop frames this similar (mean abs diff) to the previous kept one; 0 = off")
    ap.add_argument("--no-crop", action="store_true", help="save full frames, no coin crop")
    ap.add_argument("--crop-size", type=int, default=512, help="side of the saved square crop")
    ap.add_argument("--margin-min", type=float, default=1.3,
                    help="smallest crop half-size in coin radii (coin fills almost the whole image)")
    ap.add_argument("--margin-max", type=float, default=3.0,
                    help="largest crop half-size in coin radii (coin ~1/3 of the image)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--neighbor-reach", type=int, default=2,
                    help="if the coin is not found, borrow its position from a frame at most this many "
                         "frames away in the same video (0 = off)")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    rows, counts = [], Counter()
    not_found = []
    rng = np.random.default_rng(args.seed)
    method_counts = Counter()

    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        cls = class_dir.name
        for video in sorted(class_dir.iterdir()):
            if video.suffix.lower() not in VIDEO_EXT:
                continue
            m = NAME_RE.match(video.stem)
            if not m:
                print(f"  skip (bad name, expected e.g. 3R.mp4): {video}")
                continue
            coin = f"c{int(m.group(1))}"
            side = "obverse" if m.group(2).upper() == "A" else "reverse"
            vid = f"{coin}{m.group(2).upper()}"

            frames = best_frames(video, args.frames)
            if not frames:
                print(f"  could not read: {video}")
                continue
            median = float(np.median([s for s, _ in frames]))

            out_dir = dst / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            # select sharp, non-duplicate frames
            selected, prev = [], None
            for score, frame in frames:
                if score < args.blur_ratio * median:
                    continue
                t = thumb(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                if prev is not None and args.dup_thresh > 0 and np.abs(t - prev).mean() < args.dup_thresh:
                    continue
                prev = t
                selected.append((score, frame))

            # locate the coin in every selected frame
            found = [locate_coin(f) if not args.no_crop else (None, "none") for _, f in selected]
            if not args.no_crop:
                ok_idx = [i for i, (c, _) in enumerate(found) if c is not None]
                for i, (c, _) in enumerate(found):
                    if c is None and ok_idx:
                        j = min(ok_idx, key=lambda j: abs(j - i))
                        if abs(j - i) <= args.neighbor_reach:
                            found[i] = (found[j][0], "neighbor")

            kept = 0
            for (score, frame), (circle, method) in zip(selected, found):
                name = f"{cls}_{vid}_{kept:02d}.jpg"
                margin = ""
                if circle is not None:
                    margin = round(float(rng.uniform(args.margin_min, args.margin_max)), 2)
                    img = crop_coin(frame, circle, margin, args.crop_size)
                else:
                    img = resize_max(frame, args.max_side)
                    if not args.no_crop:
                        not_found.append(f"{cls}/{name}")
                cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append([f"{cls}/{name}", cls, coin, vid,
                             side, round(score, 1), method, margin])
                kept += 1
                counts[(cls, coin)] += 1
                method_counts[method] += 1
            print(f"{cls}/{video.name}: {kept} frames")

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class", "coin", "video", "side", "sharpness", "crop_method", "margin"])
        w.writerows(rows)

    print("\nSummary (images per coin):")
    for cls in sorted({c for c, _ in counts}, key=lambda c: int(c.split("_")[0]) if c.split("_")[0].isdigit() else c):
        line = ", ".join(f"{coin}={n}" for (c, coin), n in sorted(counts.items()) if c == cls)
        print(f"  {cls}: {line}  | total {sum(n for (c, _), n in counts.items() if c == cls)}")
    print(f"  total: {sum(counts.values())}")
    if not args.no_crop:
        print("\nCrop methods: " + ", ".join(f"{m}={n}" for m, n in method_counts.most_common()))
        print(f"\nCoin not found (saved uncropped, check or delete): {len(not_found)}")
        for p in not_found:
            print("  ", p)


if __name__ == "__main__":
    main()