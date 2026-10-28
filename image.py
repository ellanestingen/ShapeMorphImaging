"""
Fit a quadratic to the top or bottom outline of the red shell in each image
(a single file or a whole folder; HEIC supported).

  "bottom" = lower outline. When the camera looks slightly down on the shell,
             this is the edge CLOSEST to the camera (the arch's front profile).
  "top"    = upper outline (the far edge of the top surface in those shots).

Usage:
    python fit_quadratic.py                       # asks "top or bottom?"
    python fit_quadratic.py bottom                # no prompt
    python fit_quadratic.py /path/IMG.HEIC top    # file or folder, any order

Requires: opencv-python, numpy, pillow, pillow-heif
    pip install opencv-python numpy pillow pillow-heif
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets Pillow open .heic/.heif files

# ----------------------------- settings ------------------------------------
INPUT_PATH = "/Users/ellanestingen/Downloads/ShankarLab/2026_pics/IMG_9018.jpeg"  # a file OR a folder
EDGE = None           # "top", "bottom", or None to be asked each run (command line overrides)
SCALE_FILE = None       # path to scale.json from calibrate_scale.py; None = look next to the
                        # image, then next to this script. No file found -> results in pixels.
FOCAL_TOL = 0.05        # warn if the EXIF zoom differs from the calibration photo by more than this
OUTPUT_SUBDIR = "fits"  # annotated images are saved in INPUT_PATH/OUTPUT_SUBDIR

# "Red" detection (OpenCV hue range is 0-179; red wraps around 0).
# The shell is orange-red (hue ~0-10). The shadowed mat / interior seen beneath
# it is usually crimson (hue ~170-179), which is how the two are told apart.
H_LOW_MAX = 7     # shell hue is ~0-5; cardboard/skin are ~8+
H_HIGH_MIN = 170
S_MIN = 110     # lower = accept paler red (too low may pick up skin / cardboard)
V_MIN = 95      # lower = accept darker red (dim mat edges start to leak in)

# Dark shaded front-edge strip of the shell (used when the mat is not crimson).
EDGE_S_TEST = 200       # saturation used to decide "is there a thick dark-red interior?"
EDGE_S_MIN = 175        # saturation of the shaded edge strip that gets added (mat shadow is lower)
EDGE_MAX_FRAC = 0.25    # a strip thicker than this fraction of the shell height is
                        # really the mat / interior, so it is NOT added

# Fitting: ignore outlier edge points (foot flanges, end caps, specks).
TRIM_ENDS = 0.02      # drop this fraction of columns at each end before fitting
OUTLIER_SIGMA = 3.0   # iteratively reject points farther than this many robust sigmas

EXTS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# ---------------------------------------------------------------------------


def load_image(path):
    """Read any supported image (including HEIC) as an OpenCV BGR array."""
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil)  # apply the phone's rotation flag
        rgb = np.array(pil.convert("RGB"))
    except Exception as e:
        print(f"{Path(path).name}: {e}")
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def add_edge_strip(region, H, S, V):
    """Extend the bright region over the shell's dark, saturated front-edge strip.
    If the very-dark-red area under the shell contains a THICK blob, that area is
    the mat / interior (not an edge strip) and nothing is added."""
    ys, xs = np.nonzero(region)
    height = ys.max() - ys.min() + 1
    pad = int(EDGE_MAX_FRAC * height)
    y0, y1 = ys.min(), min(ys.max() + 2 * pad, region.shape[0])
    x0, x1 = xs.min(), xs.max() + 1
    reg = region[y0:y1, x0:x1]
    h, s_, v = H[y0:y1, x0:x1], S[y0:y1, x0:x1], V[y0:y1, x0:x1]
    bridge = cv2.dilate(reg.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0  # span thin gaps

    def attached_below(s_min):
        can = (h <= H_LOW_MAX) & (s_ >= s_min) & (v >= 70)
        allowed = (bridge | can).astype(np.uint8)
        _, lab = cv2.connectedComponents(allowed)
        ids = np.unique(lab[reg & (allowed > 0)])
        return np.isin(lab, ids[ids > 0]) & ~reg & can

    # thick-blob test (strict saturation, downscaled copy for speed)
    test = attached_below(EDGE_S_TEST)
    ds = 0.25
    small = cv2.resize(test.astype(np.uint8), None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    d = max(3, int(EDGE_MAX_FRAC * height * ds) | 1)
    thick = cv2.morphologyEx(small, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    if thick.sum() > 50:
        return region
    out = region.copy()
    out[y0:y1, x0:x1] |= attached_below(EDGE_S_MIN)
    return out


def red_mask(img_bgr):
    """Binary mask of the shell (top surface + front edge), excluding the
    darker red mat / interior visible beneath it."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    orange = (H <= H_LOW_MAX) & (S >= S_MIN) & (V >= V_MIN)
    crimson = (H >= H_HIGH_MIN) & (S >= S_MIN) & (V >= 30)

    if crimson.sum() >= 0.02 * orange.sum():
        # Usual case: the mat/interior is crimson, so hue alone isolates the shell
        # (including its shaded lower edge).
        region = orange
    else:
        # No crimson mat (e.g. brightly lit interior): fall back to splitting the
        # red region into bright (shell) and dark (interior) with Otsu's method.
        red = (orange | (crimson & (V >= V_MIN))).astype(np.uint8) * 255
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(red)
        if n <= 1:
            return None
        big = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        thr, _ = cv2.threshold(V[big].reshape(-1, 1), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        region = big & (V > thr)
        region = add_edge_strip(region, H, S, V)

    mask = region.astype(np.uint8) * 255
    # fill ink marks / glare holes, then drop specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == keep, 255, 0).astype(np.uint8)


def robust_quadratic(x, y):
    """Quadratic fit that follows the dominant arch and down-weights points a
    parabola cannot follow (curled ends, flanges, ink blobs): Tukey biweight
    IRLS, started from the central part of the outline."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    lo, hi = int(0.2 * x.size), int(0.8 * x.size)
    coef = np.polyfit(x[lo:hi], y[lo:hi], 2)
    xm, xs_ = x.mean(), x.std() + 1e-9           # condition the problem
    for _ in range(50):
        r = y - np.polyval(coef, x)
        sigma = max(1.4826 * np.median(np.abs(r - np.median(r))), 1.0)
        u = r / (OUTLIER_SIGMA * 1.5 * sigma)
        w = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if (w > 0).sum() < 10:
            break
        new = np.polyfit((x - xm) / xs_, y, 2, w=np.sqrt(w))
        # convert back to raw-x coefficients
        a2, b2, c2 = new
        new = np.array([a2 / xs_**2, b2 / xs_ - 2 * a2 * xm / xs_**2,
                        c2 - b2 * xm / xs_ + a2 * xm**2 / xs_**2])
        if np.allclose(new, coef, rtol=1e-9, atol=1e-12):
            coef = new
            break
        coef = new
    r = y - np.polyval(coef, x)
    keep = np.abs(r) <= OUTLIER_SIGMA * 1.5 * sigma
    return tuple(coef), keep


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


def find_scale(image_path):
    """Return the calibration dict from scale.json, or None if there isn't one."""
    candidates = [Path(SCALE_FILE).expanduser()] if SCALE_FILE else [
        Path(image_path).parent / "scale.json", Path(__file__).resolve().parent / "scale.json"]
    for c in candidates:
        if c.is_file():
            with open(c) as f:
                data = json.load(f)
            data["_path"] = str(c)
            return data
    return None


def exif_focal_35mm(path):
    """35 mm-equivalent focal length from EXIF, or None."""
    try:
        v = Image.open(path).getexif().get_ifd(0x8769).get(41989)
        return float(v) if v else None
    except Exception:
        return None


def scale_warnings(scale, img, path):
    """Reasons the calibration might not apply to this photo."""
    warns = []
    h, w = img.shape[:2]
    if list(scale.get("image_size", [w, h])) != [w, h]:
        warns.append(f"image size {w}x{h} differs from calibration photo {scale['image_size']}")
    f_cal = scale.get("exif", {}).get("focal_length_35mm")
    f_now = exif_focal_35mm(path)
    if f_cal and f_now and abs(f_now - f_cal) / f_cal > FOCAL_TOL:
        warns.append(f"zoom differs (35mm-equiv focal length {f_now:g} vs {f_cal:g} at calibration)")
    return warns


def analyze_image(path, edge, out_dir=None, scale="auto"):
    """Fit one image and return the results as a dict (never prints).

    scale: "auto" = look for scale.json (see find_scale), a dict from scale.json, or None.
    On failure the dict has a non-empty "error" entry.
    Keys: a, b, c (pixels), k_px, R_px, x_at_max_px, resid_px, n_points, n_used,
          and when a scale is available: mm_per_px, a_mm, c_mm, k_mm, R_mm, warnings.
    """
    path = Path(path)
    res = {"file": path.name, "edge": edge, "error": None, "warnings": []}
    img = load_image(path)
    if img is None:
        res["error"] = "could not read image"
        return res
    mask = red_mask(img)
    if mask is None:
        res["error"] = "no red region found"
        return res
    x, y = outline_points(mask, edge)
    if x.size < 3:
        res["error"] = "not enough outline points"
        return res

    # y = a x^2 + b x + c  (pixel coordinates; origin top-left, y increases downward)
    t = int(TRIM_ENDS * x.size)
    xt, yt = (x[t:x.size - t], y[t:y.size - t]) if x.size > 20 * max(t, 1) else (x, y)
    (a, b, c), keep = robust_quadratic(xt, yt)
    r = yt - np.polyval((a, b, c), xt)
    resid = 1.4826 * np.median(np.abs(r - np.median(r)))      # robust scatter about the fit (px)

    # Curvature of y = f(x):  k(x) = |f''(x)| / (1 + f'(x)^2)^(3/2),
    # with f'(x) = 2ax + b and f''(x) = 2a.  Maximum over the fitted x-range.
    xf = np.linspace(x.min(), x.max(), 500)
    k = np.abs(2 * a) / (1 + (2 * a * xf + b) ** 2) ** 1.5
    i = int(np.argmax(k))
    k_max, x_at_max = float(k[i]), float(xf[i])
    res.update(a=a, b=b, c=c, k_px=k_max, R_px=1 / k_max if k_max > 0 else float("inf"),
               x_at_max_px=x_at_max, resid_px=float(resid),
               n_points=int(x.size), n_used=int(keep.sum()))

    if scale == "auto":
        scale = find_scale(path)
    if scale:
        s = scale["mm_per_px"]            # mm per pixel
        res.update(mm_per_px=s, a_mm=a / s, c_mm=c * s, k_mm=k_max / s,
                   R_mm=s / k_max if k_max > 0 else float("inf"), resid_mm=resid * s,
                   scale_file=scale.get("_path"))
        res["warnings"] = scale_warnings(scale, img, path)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pts = np.stack([xf, a * xf**2 + b * xf + c], axis=1).round().astype(np.int32).reshape(-1, 1, 2)
        annotated = img.copy()
        cv2.polylines(annotated, [pts], False, (255, 0, 0), thickness=3)   # blue (BGR)
        cv2.imwrite(str(out_dir / f"{path.stem}_fit_{edge}.png"), annotated)
        cv2.imwrite(str(out_dir / f"{path.stem}_mask.png"), mask)           # for checking the mask
    return res


def process(path, edge, out_dir):
    """Command-line wrapper: fit one image and print the results."""
    path = Path(path)
    res = analyze_image(path, edge, out_dir)
    if res["error"]:
        print(f"{path.name}: {res['error']}, skipping")
        return
    print(f"{path.name} ({edge}):  a = {res['a']:.6e}   b = {res['b']:.6e}   c = {res['c']:.6e}")
    print(f"    max curvature k = {res['k_px']:.6e} 1/px  at x = {res['x_at_max_px']:.1f} px  "
          f"(radius of curvature = {res['R_px']:.1f} px)")
    if "mm_per_px" not in res:
        print("    (no scale.json found: results are in pixels)")
    else:
        print(f"    in mm  (scale {res['mm_per_px']:.6f} mm/px):  a = {res['a_mm']:.6e} 1/mm   "
              f"b = {res['b']:.6e}   c = {res['c_mm']:.6e} mm")
        print(f"    max curvature k = {res['k_mm']:.6e} 1/mm   "
              f"radius of curvature = {res['R_mm']:.3f} mm")
        for w in res["warnings"]:
            print(f"    WARNING: {w}. The scale may not apply to this photo.")


def main():
    # Optional command-line args, in any order: "top"/"bottom" and/or a path
    edge = EDGE.lower() if EDGE else None
    target = INPUT_PATH
    for arg in sys.argv[1:]:
        if arg.lower() in ("top", "bottom"):
            edge = arg.lower()
        else:
            target = arg
    while edge not in ("top", "bottom"):
        edge = input("Fit the top or bottom outline? [top/bottom]: ").strip().lower()

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
    