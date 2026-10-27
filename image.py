"""
Fit a quadratic to the top or bottom outline of a red object in each image
of a folder.

Usage:
    python fit_quadratic.py                # uses the settings below
    python fit_quadratic.py top
    python fit_quadratic.py bottom

Requires: opencv-python, numpy   (pip install opencv-python numpy)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- settings ------------------------------------
INPUT_PATH = "/Users/ellanestingen/Downloads/ShankarLab/2026_pics/IMG_9018.jpeg"  # a file OR a folder
EDGE = "top"          # "top" or "bottom" (can be overridden on the command line)
OUTPUT_SUBDIR = "fits"  # annotated images are saved in INPUT_PATH/OUTPUT_SUBDIR

# HSV thresholds for "red" (OpenCV hue range is 0-179; red wraps around 0).
# Raise S_MIN / V_MIN if the darker red mat is being picked up;
# lower them if parts of the shape are missing.
H_LOW_MAX = 10
H_HIGH_MIN = 170
S_MIN = 120
V_MIN = 100

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# ---------------------------------------------------------------------------


def red_mask(img_bgr):
    """Binary mask of the red shape (largest red connected component)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lo = cv2.inRange(hsv, (0, S_MIN, V_MIN), (H_LOW_MAX, 255, 255))
    hi = cv2.inRange(hsv, (H_HIGH_MIN, S_MIN, V_MIN), (179, 255, 255))
    mask = cv2.bitwise_or(lo, hi)

    # Clean up speckle and small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest blob
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def outline_points(mask, edge):
    """For every column containing mask pixels, return the top or bottom y."""
    xs, ys = [], []
    for x in range(mask.shape[1]):
        col = np.flatnonzero(mask[:, x])
        if col.size == 0:
            continue
        xs.append(x)
        ys.append(col.min() if edge == "top" else col.max())
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def process(path, edge, out_dir):
    img = cv2.imread(str(path))
    if img is None:
        print(f"{path.name}: could not read image, skipping")
        return

    mask = red_mask(img)
    if mask is None:
        print(f"{path.name}: no red region found, skipping")
        return

    x, y = outline_points(mask, edge)
    if x.size < 3:
        print(f"{path.name}: not enough outline points, skipping")
        return

    # y = a x^2 + b x + c  (pixel coordinates; origin top-left, y increases downward)
    a, b, c = np.polyfit(x, y, 2)

    # Draw the fit in blue (BGR) over the original image
    xf = np.linspace(x.min(), x.max(), 500)
    yf = a * xf**2 + b * xf + c
    pts = np.stack([xf, yf], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
    annotated = img.copy()
    cv2.polylines(annotated, [pts], False, (255, 0, 0), thickness=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{path.stem}_fit_{edge}.png"), annotated)
    cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), mask)  # for checking the mask

    # Curvature of y = f(x):  k(x) = |f''(x)| / (1 + f'(x)^2)^(3/2),
    # with f'(x) = 2ax + b and f''(x) = 2a.  Maximum over the fitted x-range.
    k = np.abs(2 * a) / (1 + (2 * a * xf + b) ** 2) ** 1.5
    i = np.argmax(k)
    k_max, x_at_max = k[i], xf[i]

    print(f"{path.name} ({edge}):  a = {a:.6e}   b = {b:.6e}   c = {c:.6e}")
    print(f"    max curvature k = {k_max:.6e} 1/px  at x = {x_at_max:.1f} px  "
          f"(radius of curvature = {1 / k_max:.1f} px)")


def main():
    # Optional command-line args, in any order: "top"/"bottom" and/or a path
    edge = EDGE.lower()
    target = INPUT_PATH
    for arg in sys.argv[1:]:
        if arg.lower() in ("top", "bottom"):
            edge = arg.lower()
        else:
            target = arg
    if edge not in ("top", "bottom"):
        sys.exit('EDGE must be "top" or "bottom".')

    target = Path(target).expanduser()
    if target.is_file():
        files = [target]
        out_dir = target.parent / OUTPUT_SUBDIR
    elif target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in EXTS)
        out_dir = target / OUTPUT_SUBDIR
        if not files:
            sys.exit(f"No images found in {target}")
    else:
        sys.exit(f"Path not found: {target}")

    for f in files:
        process(f, edge, out_dir)
    print(f"\nAnnotated images and masks saved to: {out_dir}")


if __name__ == "__main__":
    main()