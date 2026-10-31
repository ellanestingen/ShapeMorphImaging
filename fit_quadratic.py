"""
Fit a parabola to the edge of the red shell in a photo and report its curvature.

The shell is a bright red curved strip photographed against a plain background, usually
resting on a darker red mat. For each photo this script
  1. builds a mask of the shell (find_shell_mask),
  2. takes one outline of that mask: its "bottom" or its "top" edge,
  3. fits  y = a*x^2 + b*x + c  to that outline (robust to curled ends and specks),
  4. reports the maximum curvature  k = 2|a|  and the radius  R = 1/k,
     in pixels and, if a scale.json from calibrate_scale.py is found, also in mm.

Which edge:
  "bottom"  the lower outline. When the camera looks slightly down on the shell, this is
            the edge CLOSEST to the camera (the front profile of the arch).
  "top"     the upper outline (in those shots, the far edge of the top surface).

Usage:
    python3 fit_quadratic.py                           # photo/folder from INPUT_PATH, asks top/bottom
    python3 fit_quadratic.py bottom                    # no question
    python3 fit_quadratic.py /path/to/photo.HEIC top   # a photo or a folder, in any order

For every photo, a "fits" folder next to it receives
    <name>_fit_<edge>.png   the photo with the fitted curve drawn in blue
    <name>_mask.png         the detected shell (look here first if a fit looks wrong)

Image coordinates: x points right and y points DOWN, so the fitted a, b, c are in that frame.

analyze_prints.py imports this file; keep the two in the same folder.
Requires: opencv-python, numpy, pillow, pillow-heif
    python3 -m pip install opencv-python numpy pillow pillow-heif
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()  # lets Pillow open .heic / .heif photos

# =============================== SETTINGS ====================================
INPUT_PATH = "/Users/ellanestingen/Downloads/ShankarLab/CurvatureImages"  # a photo OR a folder
EDGE = None                 # "top", "bottom", or None to be asked each run (a command-line word wins)
SCALE_FILE = None           # path to scale.json; None = look next to the photo, then next to this script
OUTPUT_SUBDIR = "fits"      # folder (next to the photos) for the annotated images and masks
IMAGE_EXTENSIONS = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
EXTS = IMAGE_EXTENSIONS    # old name, kept so older scripts that import it still work

# --- Colour rules for finding the shell ----------------------------------------
# OpenCV HSV: hue runs 0-179 and red wraps around 0. Saturation and value run 0-255.
SHELL_HUE_MAX = 7           # the shell is orange-red (hue ~0-5); cardboard and skin are ~8 and up
CRIMSON_HUE_MIN = 170       # the shadowed mat / interior is usually crimson (hue ~170-179)
SAT_MIN = 110               # lower = accept paler red (too low picks up skin and cardboard)
VAL_MIN = 95                # lower = accept darker red (too low lets dim mat edges leak in)
CRIMSON_VAL_MIN = 30        # how dark a crimson pixel may be and still count as "mat"
CRIMSON_FRACTION = 0.02     # the mat counts as crimson if crimson pixels are at least this
                            # fraction of the shell-coloured pixels

# --- Shaded front-edge strip (only needed when the mat is NOT crimson) ---------
STRIP_SAT_MIN = 175         # saturation of the dark edge strip that gets added to the shell
STRIP_SAT_TEST = 200        # stricter saturation used to decide whether the dark red area is
                            # a thin edge strip (add it) or a thick interior (leave it out)
STRIP_VAL_MIN = 70          # darkest value the edge strip may have
STRIP_MAX_FRAC = 0.25       # an edge strip is thinner than this fraction of the shell's height

# --- Mask clean-up (kernel sizes, in pixels) -----------------------------------
RAW_SPECK_KERNEL = 7        # removes speckle from the raw red pixels
HOLE_FILL_KERNEL = 31       # fills ink marks and glare spots inside the shell
SPECK_KERNEL = 9            # removes thin leftovers after filling

# --- Rejecting photos with no usable shell ---------------------------------------
MAX_SHELL_AREA_FRAC = 0.25  # a "shell" bigger than this fraction of the photo is probably a hand
MAX_BORDER_SIDES = 1        # ...and so is one touching more than this many photo borders

# --- Fitting -------------------------------------------------------------------
SLOPE_LIMIT = 1.0           # fit only the part of the outline flatter than this (|dy/dx|, 1 = 45 deg)
                            # around the apex; None = use the whole outline. Steep legs cannot be
                            # followed by a parabola and would pull the fit off the arch.
MIN_WINDOW_FRAC = 0.25      # ...but never fit less than this fraction of the shell's width
JUMP_FRAC = 0.03            # the outline "jumps" (a different part of the shell begins) where it
                            # moves by more than this fraction of the shell's width between
                            # neighbouring columns; the fit never crosses a jump or a gap
TRIM_ENDS = 0.02            # fraction of outline columns ignored at each end of the shell
TUKEY_CUTOFF = 4.5          # points further than this many robust sigmas get zero weight
MIN_SIGMA_PX = 1.0          # floor for the robust sigma, in pixels
MIN_OUTLINE_POINTS = 10     # fewer outline columns than this -> the photo is rejected

# --- Fit-quality flag (a fit is marked "check" if any of these is true) ----------
CHECK_MIN_KEPT = 0.85       # fewer than this fraction of the fitted points agree with the curve
CHECK_MAX_SCATTER = 0.01    # scatter about the curve bigger than this fraction of the shell width
CHECK_MIN_WINDOW = 0.40     # less than this fraction of the shell's width could be fitted

# --- Scale check ---------------------------------------------------------------
FOCAL_TOL = 0.05            # warn if the EXIF zoom differs from the calibration photo by more
# =============================================================================

EXIF_FIELDS = {41989: "focal_length_35mm", 41988: "digital_zoom_ratio", 37386: "focal_length_mm"}


# ----------------------------------------------------------------------------
# Reading photos
# ----------------------------------------------------------------------------
def load_image(path):
    """Read a photo (HEIC included) as an OpenCV BGR array, honouring the phone's rotation flag.
    Raises an exception if the file cannot be read."""
    pil = ImageOps.exif_transpose(Image.open(path))
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def read_exif(path):
    """A few camera fields from the photo's EXIF data ({} if there are none)."""
    try:
        sub = Image.open(path).getexif().get_ifd(0x8769)  # the Exif sub-directory
        return {name: float(sub[tag]) for tag, name in EXIF_FIELDS.items() if tag in sub}
    except Exception:
        return {}


# ----------------------------------------------------------------------------
# Finding the shell
# ----------------------------------------------------------------------------
def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _largest_component(binary):
    """Boolean mask of the largest connected blob in a 0/255 image (None if it is empty)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return None
    return labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])


def add_edge_strip(region, H, S, V):
    """Extend the bright shell region over its dark, saturated front-edge strip.

    If the dark-red area attached below the shell contains a THICK blob, that area is the
    mat / interior rather than an edge strip, and nothing is added."""
    ys, xs = np.nonzero(region)
    height = ys.max() - ys.min() + 1
    pad = int(STRIP_MAX_FRAC * height)
    y0, y1 = ys.min(), min(ys.max() + 2 * pad, region.shape[0])   # search window
    x0, x1 = xs.min(), xs.max() + 1
    shell = region[y0:y1, x0:x1]
    h, s, v = H[y0:y1, x0:x1], S[y0:y1, x0:x1], V[y0:y1, x0:x1]
    bridged = cv2.dilate(shell.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0  # spans thin gaps

    def dark_red_below_shell(sat_min):
        dark_red = (h <= SHELL_HUE_MAX) & (s >= sat_min) & (v >= STRIP_VAL_MIN)
        reachable = (bridged | dark_red).astype(np.uint8)
        _, labels = cv2.connectedComponents(reachable)
        touching = np.unique(labels[shell & (reachable > 0)])
        return np.isin(labels, touching[touching > 0]) & ~shell & dark_red

    # Thick-blob test on a quarter-size copy (fast): does an opening that is as wide as
    # the thickest allowed strip leave anything behind?
    scale = 0.25
    small = cv2.resize(dark_red_below_shell(STRIP_SAT_TEST).astype(np.uint8), None,
                       fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kernel = max(3, int(STRIP_MAX_FRAC * height * scale) | 1)
    thick_part = cv2.morphologyEx(small, cv2.MORPH_OPEN, _ellipse(kernel))
    if thick_part.sum() > 50:
        return region

    grown = region.copy()
    grown[y0:y1, x0:x1] |= dark_red_below_shell(STRIP_SAT_MIN)
    return grown


def _bright_region(H, S, V, shell_coloured, crimson):
    """Shell pixels when the mat is not crimson: split the red area into a bright part
    (the shell) and a darker part (the interior) with Otsu's threshold on brightness."""
    red = (shell_coloured | (crimson & (V >= VAL_MIN))).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, _ellipse(RAW_SPECK_KERNEL))
    blob = _largest_component(red)
    if blob is None:
        return None
    threshold, _ = cv2.threshold(V[blob].reshape(-1, 1), 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return add_edge_strip(blob & (V > threshold), H, S, V)


def find_shell_mask(img_bgr):
    """0/255 mask of the shell (top surface + front edge), without the darker red mat or
    interior seen beneath it. Returns None if no shell is found.

    Two cases, chosen automatically:
      * the mat is crimson (hue wraps to ~170-179): the shell's orange-red hue alone separates them;
      * the mat is not crimson (e.g. brightly lit): fall back to a brightness split."""
    H, S, V = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    shell_coloured = (H <= SHELL_HUE_MAX) & (S >= SAT_MIN) & (V >= VAL_MIN)
    crimson = (H >= CRIMSON_HUE_MIN) & (S >= SAT_MIN) & (V >= CRIMSON_VAL_MIN)

    if crimson.sum() >= CRIMSON_FRACTION * shell_coloured.sum():
        region = shell_coloured
    else:
        region = _bright_region(H, S, V, shell_coloured, crimson)
        if region is None:
            return None

    mask = region.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(HOLE_FILL_KERNEL))   # ink marks, glare
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ellipse(SPECK_KERNEL))        # specks
    blob = _largest_component(mask)
    return None if blob is None else blob.astype(np.uint8) * 255


# ----------------------------------------------------------------------------
# Fitting the outline
# ----------------------------------------------------------------------------
def shell_problem(mask):
    """Why this mask cannot be a shell (a hand in front of the camera, say), or None if it is fine."""
    h, w = mask.shape
    if np.count_nonzero(mask) > MAX_SHELL_AREA_FRAC * h * w:
        return "red area is too large to be the shell (a hand or other object in the way?)"
    sides = [mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()]
    if sum(sides) > MAX_BORDER_SIDES:
        return "red area runs off the edges of the photo (a hand or other object in the way?)"
    return None


def outline_points(mask, edge):
    """For every image column that contains shell, the y of its top or bottom shell pixel.
    Returns (x, y) as float arrays."""
    columns = np.flatnonzero(mask.any(axis=0))
    if edge == "top":
        y = mask[:, columns].argmax(axis=0)
    else:
        y = mask.shape[0] - 1 - mask[::-1, columns].argmax(axis=0)
    return columns.astype(float), y.astype(float)


def fit_window(x, y):
    """Which outline points to fit: a stretch around the apex that is flatter than SLOPE_LIMIT.

    A parabola cannot follow near-vertical legs (the outline barely moves in x while y changes
    a lot), and such points would bend the whole fit. Returns a boolean array over the points."""
    n = x.size
    everything = np.ones(n, bool)
    if SLOPE_LIMIT is None or n < 40:
        return everything

    wide = max(5, n // 20) | 1                     # median-smooth the outline to suppress spikes
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(
        np.pad(y, wide // 2, mode="edge"), wide), axis=1)
    centre = slice(int(0.2 * n), int(0.8 * n))
    arch = np.polyfit(x[centre], y[centre], 2)[0] >= 0      # a >= 0: highest point is the apex
    apex = int(np.argmin(smooth) if arch else np.argmax(smooth))

    steep = np.abs(np.gradient(smooth, x)) > SLOPE_LIMIT
    run = max(3, n // 50)                          # a "wall" must be steep for a run of columns
    wall = np.convolve(steep.astype(int), np.ones(run, int), mode="same") >= run
    # barrier[i] is True if the outline is broken between point i and point i+1
    barrier = (np.abs(np.diff(y)) > max(15.0, JUMP_FRAC * np.ptp(x))) | (np.diff(x) > 1.5)
    lo, hi = apex, apex
    while lo > 0 and not wall[lo - 1] and not barrier[lo - 1]:
        lo -= 1
    while hi < n - 1 and not wall[hi + 1] and not barrier[hi]:
        hi += 1

    need = int(MIN_WINDOW_FRAC * n)                # keep a minimum amount of outline,
    while hi - lo + 1 < need:                      # but never across a jump or gap
        grew = False
        if lo > 0 and not barrier[lo - 1]:
            lo -= 1
            grew = True
        if hi < n - 1 and not barrier[hi]:
            hi += 1
            grew = True
        if not grew:
            break
    chosen = np.zeros(n, bool)
    chosen[lo:hi + 1] = True
    return chosen


def robust_quadratic(x, y):
    """Fit y = a x^2 + b x + c so that it follows the dominant arch and ignores points a
    parabola cannot follow (curled ends, flanges, ink blobs).

    Tukey-biweight iteratively reweighted least squares, started from the central 60% of the
    outline. Returns ((a, b, c), keep), where keep marks the points that still count."""
    order = np.argsort(x)
    x, y = x[order], y[order]
    centre = slice(int(0.2 * x.size), int(0.8 * x.size))
    coef = np.polyfit(x[centre], y[centre], 2)

    for _ in range(50):
        residual = y - np.polyval(coef, x)
        sigma = max(1.4826 * np.median(np.abs(residual - np.median(residual))), MIN_SIGMA_PX)
        u = residual / (TUKEY_CUTOFF * sigma)
        weight = np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)
        if np.count_nonzero(weight) < MIN_OUTLINE_POINTS:
            break
        new = np.polyfit(x, y, 2, w=np.sqrt(weight))
        converged = np.allclose(new, coef, rtol=1e-9, atol=1e-12)
        coef = new
        if converged:
            break

    keep = np.abs(y - np.polyval(coef, x)) <= TUKEY_CUTOFF * sigma
    return tuple(coef), keep


def max_curvature(a, b, x_min, x_max):
    """Largest curvature of y = a x^2 + b x + c on [x_min, x_max], as (k, x_at_max).

    Curvature k(x) = |y''| / (1 + y'^2)^(3/2) with y' = 2ax + b and y'' = 2a. It is largest
    at the vertex (where y' = 0, giving k = 2|a|), or at the nearer end if the vertex lies
    outside the photographed part of the shell."""
    if a == 0:
        return 0.0, 0.5 * (x_min + x_max)
    x = float(np.clip(-b / (2 * a), x_min, x_max))
    return 2 * abs(a) / (1 + (2 * a * x + b) ** 2) ** 1.5, x


# ----------------------------------------------------------------------------
# Scale (pixels -> mm)
# ----------------------------------------------------------------------------
def find_scale(image_path):
    """The calibration dict from scale.json, or None if there is no such file."""
    if SCALE_FILE:
        candidates = [Path(SCALE_FILE).expanduser()]
    else:
        candidates = [Path(image_path).parent / "scale.json",
                      Path(__file__).resolve().parent / "scale.json"]
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate) as f:
                scale = json.load(f)
            if "mm_per_px" not in scale:
                raise ValueError(f"{candidate} has no 'mm_per_px' entry")
            scale["_path"] = str(candidate)
            return scale
    return None


def scale_warnings(scale, image_size, path):
    """Reasons the calibration might not apply to this photo (empty list if none)."""
    warnings = []
    width, height = image_size
    if list(scale.get("image_size", [width, height])) != [width, height]:
        warnings.append(f"image size {width}x{height} differs from the calibration photo "
                        f"{scale['image_size']}")
    focal_cal = scale.get("exif", {}).get("focal_length_35mm")
    focal_now = read_exif(path).get("focal_length_35mm")
    if focal_cal and focal_now and abs(focal_now - focal_cal) / focal_cal > FOCAL_TOL:
        warnings.append(f"zoom differs (35mm-equivalent focal length {focal_now:g} vs "
                        f"{focal_cal:g} at calibration)")
    return warnings


# ----------------------------------------------------------------------------
# One photo -> results
# ----------------------------------------------------------------------------
def save_overlay(img, mask, curve, edge, stem, out_dir):
    """Write the photo with the fitted curve (blue) and the mask to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = curve.round().astype(np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.polylines(overlay, [points], False, (255, 0, 0), thickness=3)  # BGR: blue
    cv2.imwrite(str(out_dir / f"{stem}_fit_{edge}.png"), overlay)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)


def analyze_image(path, edge, out_dir=None, scale="auto"):
    """Fit one photo and return the results as a dict (nothing is printed).

    edge     "top" or "bottom".
    out_dir  where to write the annotated image and mask (None = write nothing).
    scale    "auto" = look for scale.json; or a dict loaded from one; or None = pixels only.

    The dict always has "file", "edge", "error" (None on success), "quality" and "warnings".
    On success it also has, in pixels: a, b, c, k_px, R_px, x_at_max_px, resid_px (scatter of
    the fitted outline about the curve), n_points, n_fitted, n_used, fit_x_min_px, fit_x_max_px,
    window_frac, quality_note; and, when there is a scale: mm_per_px, a_mm, c_mm, k_mm, R_mm,
    resid_mm, scale_file.
    "quality" is "ok" or "check" (look at the overlay: see CHECK_* settings)."""
    path = Path(path)
    result = {"file": path.name, "edge": edge, "error": None, "quality": "", "warnings": []}

    try:
        img = load_image(path)
    except Exception as exc:
        result["error"] = f"could not read image ({exc})"
        return result
    mask = find_shell_mask(img)
    if mask is None:
        result["error"] = "no red shell found"
        return result
    problem = shell_problem(mask)
    if problem:
        result["error"] = problem
        return result
    x, y = outline_points(mask, edge)
    if x.size < MIN_OUTLINE_POINTS:
        result["error"] = "shell outline too short"
        return result

    # Fit only the stretch of the outline a parabola can follow (see fit_window)
    trim = int(TRIM_ENDS * x.size)
    x, y = x[trim:x.size - trim], y[trim:x.size - trim]
    chosen = fit_window(x, y)
    x_fit, y_fit = x[chosen], y[chosen]
    (a, b, c), keep = robust_quadratic(x_fit, y_fit)
    misfit = y_fit - np.polyval((a, b, c), x_fit)
    scatter = float(1.4826 * np.median(np.abs(misfit - np.median(misfit))))
    x_lo, x_hi = float(x_fit.min()), float(x_fit.max())
    k, x_at_max = max_curvature(a, b, x_lo, x_hi)

    width = float(x.max() - x.min())
    kept = keep.sum() / keep.size
    window_frac = (x_hi - x_lo) / width if width > 0 else 1.0
    notes = []
    if kept < CHECK_MIN_KEPT:
        notes.append(f"only {kept:.0%} of the fitted outline agrees with the curve")
    if scatter > CHECK_MAX_SCATTER * width:
        notes.append(f"outline scatters {scatter:.0f}px about the curve")
    if window_frac < CHECK_MIN_WINDOW:
        notes.append(f"only the central {window_frac:.0%} of the shell was fitted (steep sides)")

    result.update(a=a, b=b, c=c, k_px=k, R_px=1 / k if k > 0 else float("inf"),
                  x_at_max_px=x_at_max, resid_px=scatter, n_points=int(x.size),
                  n_fitted=int(x_fit.size), n_used=int(keep.sum()),
                  fit_x_min_px=x_lo, fit_x_max_px=x_hi, window_frac=window_frac,
                  quality="check" if notes else "ok", quality_note="; ".join(notes))

    if scale == "auto":
        scale = find_scale(path)
    if scale:
        s = scale["mm_per_px"]  # mm per pixel: lengths x s, curvature / s
        result.update(mm_per_px=s, a_mm=a / s, c_mm=c * s, k_mm=k / s,
                      R_mm=s / k if k > 0 else float("inf"), resid_mm=scatter * s,
                      scale_file=scale.get("_path"))
        result["warnings"] = scale_warnings(scale, (img.shape[1], img.shape[0]), path)

    if out_dir is not None:
        xs = np.linspace(x_lo, x_hi, 500)             # draw the curve where it was fitted
        curve = np.stack([xs, a * xs**2 + b * xs + c], axis=1)
        save_overlay(img, mask, curve, edge, path.stem, out_dir)
    return result


def print_result(result):
    """Print one analyze_image() result for the command line."""
    name = result["file"]
    if result["error"]:
        print(f"{name}: {result['error']}, skipped")
        return
    print(f"{name} ({result['edge']}):  a = {result['a']:.6e}   b = {result['b']:.6e}   "
          f"c = {result['c']:.6e}")
    print(f"    max curvature k = {result['k_px']:.6e} 1/px  at x = {result['x_at_max_px']:.1f} px  "
          f"(radius of curvature = {result['R_px']:.1f} px)")
    if result["quality"] == "check":
        print(f"    CHECK THIS FIT: {result['quality_note']}")
    if "mm_per_px" not in result:
        print("    (no scale.json found: results are in pixels)")
        return
    print(f"    in mm (scale {result['mm_per_px']:.6f} mm/px):  a = {result['a_mm']:.6e} 1/mm   "
          f"b = {result['b']:.6e}   c = {result['c_mm']:.6e} mm")
    print(f"    max curvature k = {result['k_mm']:.6e} 1/mm   "
          f"radius of curvature = {result['R_mm']:.3f} mm")
    for warning in result["warnings"]:
        print(f"    WARNING: {warning}. The scale may not apply to this photo.")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def main():
    edge = EDGE.lower() if EDGE else None
    target = INPUT_PATH
    for arg in sys.argv[1:]:                      # "top"/"bottom" and/or a path, any order
        if arg.lower() in ("top", "bottom"):
            edge = arg.lower()
        else:
            target = arg
    while edge not in ("top", "bottom"):
        edge = input("Fit the top or bottom outline? [top/bottom]: ").strip().lower()

    target = Path(target.strip().strip("'\"")).expanduser()
    if target.is_file():
        photos, out_dir = [target], target.parent / OUTPUT_SUBDIR
    elif target.is_dir():
        photos = sorted(p for p in target.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith("."))
        out_dir = target / OUTPUT_SUBDIR
        if not photos:
            sys.exit(f"No images found in {target}")
    else:
        sys.exit(f"Path not found: {target}")

    for photo in photos:
        print_result(analyze_image(photo, edge, out_dir))
    print(f"\nAnnotated images and masks saved to: {out_dir}")


if __name__ == "__main__":
    main()