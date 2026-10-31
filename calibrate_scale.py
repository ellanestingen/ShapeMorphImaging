"""
Calibrate pixels -> millimetres from a photo containing a ruler (or any known length).

You click two points on the image, type the real distance between them in mm, and the
scale is saved to a small JSON file (scale.json) that other scripts can read.

Usage:
    python3 calibrate_scale.py                         # asks for the image path
    python3 calibrate_scale.py /path/IMG_9162.HEIC     # opens the image
    python3 calibrate_scale.py IMG.HEIC --out my_scale.json

    # non-interactive (no window): give the pixel coordinates and distance yourself
    python3 calibrate_scale.py IMG.HEIC --points 520,2455,520,215 --distance 70

How the clicking works (for precision):
    1. The whole photo is shown shrunk to fit your screen. Click roughly where the first
       point is.
    2. A magnified, full-resolution close-up of that spot opens. Click the exact spot.
       (Press r to redo that point, Esc to quit.)
    3. Repeat for the second point.
    Pick two marks FAR APART (e.g. the 10 mm and 80 mm marks): the further apart, the
    smaller the relative error from your click accuracy.

Keep this file in the same folder as fit_quadratic.py (it reuses its photo loader).
Requires: opencv-python (the regular build, NOT -headless: it needs to open a window),
          numpy, pillow, pillow-heif
    python3 -m pip install opencv-python numpy pillow pillow-heif
"""

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import cv2

from fit_quadratic import load_image, read_exif

DISPLAY_MAX = 900      # longest side (px) of the overview window
ZOOM_HALF = 100        # close-up shows a (2*ZOOM_HALF)^2 px full-resolution patch...
ZOOM_FACTOR = 5        # ...magnified this many times


# ----------------------------------------------------------------------------
def pick_point(img, label):
    """Two-stage click: coarse on the overview, then fine on a magnified patch.
    Returns (x, y) in full-resolution pixel coordinates, or None if the user quits."""
    h, w = img.shape[:2]
    scale = min(1.0, DISPLAY_MAX / max(h, w))
    overview = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    while True:
        # ---- stage 1: coarse click on the whole image
        coarse = click_once(overview, f"{label}: click roughly (Esc = quit)")
        if coarse is None:
            return None
        cx, cy = coarse[0] / scale, coarse[1] / scale

        # ---- stage 2: precise click on a magnified full-resolution patch
        x0 = int(min(max(cx - ZOOM_HALF, 0), max(w - 2 * ZOOM_HALF, 0)))
        y0 = int(min(max(cy - ZOOM_HALF, 0), max(h - 2 * ZOOM_HALF, 0)))
        patch = img[y0:y0 + 2 * ZOOM_HALF, x0:x0 + 2 * ZOOM_HALF]
        big = cv2.resize(patch, None, fx=ZOOM_FACTOR, fy=ZOOM_FACTOR,
                         interpolation=cv2.INTER_CUBIC)
        fine = click_once(big, f"{label}: click EXACT spot (r = redo, Esc = quit)",
                          allow_redo=True)
        if fine is None:
            return None
        if fine == "redo":
            continue
        return x0 + fine[0] / ZOOM_FACTOR, y0 + fine[1] / ZOOM_FACTOR


def click_once(image, title, allow_redo=False):
    """Show an image and wait for one left click. Returns (x, y) in that image's pixel
    coordinates, 'redo' if r was pressed (when allowed), or None on Esc."""
    state = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["pt"] = (x, y)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)
    cv2.imshow(title, image)
    result = None
    while True:
        key = cv2.waitKey(30) & 0xFF
        if state["pt"] is not None:
            # brief visual confirmation of where you clicked
            shown = image.copy()
            cv2.drawMarker(shown, state["pt"], (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
            cv2.imshow(title, shown)
            cv2.waitKey(250)
            result = state["pt"]
            break
        if key == 27:                           # Esc
            break
        if allow_redo and key in (ord("r"), ord("R")):
            result = "redo"
            break
    cv2.destroyWindow(title)
    cv2.waitKey(1)
    return result


def show_result(img, p1, p2):
    """Draw the chosen points and the line between them for a final visual check."""
    out = img.copy()
    t = max(2, img.shape[1] // 600)
    for p in (p1, p2):
        cv2.drawMarker(out, (int(p[0]), int(p[1])), (0, 255, 0), cv2.MARKER_CROSS, 8 * t, t)
    cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), t)
    scale = min(1.0, DISPLAY_MAX / max(out.shape[:2]))
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow("Chosen points - press any key", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def compute_scale(p1, p2, distance_mm):
    pix = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if pix < 1e-9:
        raise ValueError("The two points are identical.")
    if distance_mm <= 0:
        raise ValueError("Distance must be positive.")
    return distance_mm / pix, pix


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Calibrate pixels -> mm from a ruler photo.")
    ap.add_argument("image", nargs="?", help="photo containing the ruler")
    ap.add_argument("--out", help="output JSON (default: scale.json next to the image)")
    ap.add_argument("--points", help="x1,y1,x2,y2 in full-resolution pixels (skips clicking)")
    ap.add_argument("--distance", type=float, help="real distance between the points, in mm")
    args = ap.parse_args()

    path = args.image or input("Path to the ruler photo: ")
    path = Path(path.strip().strip("'\"").replace("\\ ", " ")).expanduser()
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    try:
        img = load_image(path)
    except Exception as exc:
        sys.exit(f"Could not read {path.name}: {exc}")
    h, w = img.shape[:2]
    print(f"Loaded {path.name}: {w} x {h} px (after applying phone rotation)")

    if args.points:
        try:
            x1, y1, x2, y2 = [float(v) for v in args.points.split(",")]
        except ValueError:
            sys.exit("--points must be four numbers: x1,y1,x2,y2")
        p1, p2 = (x1, y1), (x2, y2)
    else:
        print("Click two marks FAR APART on the ruler (e.g. the 10 mm and 80 mm marks).")
        p1 = pick_point(img, "Point 1 of 2")
        if p1 is None:
            sys.exit("Cancelled.")
        p2 = pick_point(img, "Point 2 of 2")
        if p2 is None:
            sys.exit("Cancelled.")
        show_result(img, p1, p2)

    pix = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    print(f"Point 1: ({p1[0]:.1f}, {p1[1]:.1f})   Point 2: ({p2[0]:.1f}, {p2[1]:.1f})")
    print(f"Pixel distance: {pix:.1f} px")

    if args.distance is not None:
        d_mm = args.distance
    else:
        while True:
            try:
                d_mm = float(input("Real distance between the two points (mm): "))
                break
            except ValueError:
                print("Please type a number, e.g. 70")

    try:
        mm_per_px, pix = compute_scale(p1, p2, d_mm)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    print(f"\nScale: {mm_per_px:.6f} mm/px   ({1 / mm_per_px:.3f} px/mm)")
    if pix < 0.25 * max(w, h):
        print("Note: the points are fairly close together. Points further apart give a "
              "more accurate scale.")

    out_path = Path(args.out) if args.out else path.parent / "scale.json"
    data = {
        "mm_per_px": mm_per_px,
        "px_per_mm": 1 / mm_per_px,
        "distance_mm": d_mm,
        "pixel_distance": pix,
        "point1": list(p1),
        "point2": list(p2),
        "image_file": path.name,
        "image_size": [w, h],
        "exif": read_exif(path),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "note": "Valid only for photos taken with the same camera, zoom and distance "
                "to the object plane as the calibration photo.",
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()