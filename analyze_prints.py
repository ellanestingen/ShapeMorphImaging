"""
Batch analysis of the shell photos in one folder.

File names must look like   totalheight_speed_printcount_side_sidecount.HEIC
    2_80_1_h_1.HEIC   ->  total height 2 mm, speed 80 mm/s, print #1, 'h' measurement, side #1
    (the trailing side count is optional; the direction letter is h or w)

For every image the script
  1. parses the print parameters from the file name,
  2. fits the near edge with fit_quadratic.py and gets the curvature,
  3. converts pixels to mm with the scale.json made by calibrate_scale.py,
and then writes, into  <folder>/analysis/ :
  results.csv                one row per image (parameters + fit + curvature + warnings)
  plot_speed_<speed>.png     curvature vs total height, height (h) vs width (w) colour-coded,
                             one plot per print speed
  fit_check_sheet.png        thumbnails of every fit so you can check them by eye
  fits/                      full-size annotated image and mask for each photo

Usage:
    python3 analyze_prints.py                       # uses FOLDER below
    python3 analyze_prints.py /path/to/folder
    python3 analyze_prints.py /path/to/folder --edge bottom --quantity radius

Keep this file in the same folder as fit_quadratic.py (it imports it).
Requires: opencv-python, numpy, pillow, pillow-heif, matplotlib
    python3 -m pip install opencv-python numpy pillow pillow-heif matplotlib
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt
import numpy as np

import fit_quadratic as fq

# ----------------------------- settings ------------------------------------
FOLDER = "/Users/ellanestingen/Downloads/ShankarLab/CurvatureImages"
EDGE = "bottom"          # which outline to fit: "bottom" = edge closest to the camera
SCALE_FILE = None        # None = scale.json inside FOLDER (or next to the scripts)
OUTPUT_SUBDIR = "analysis"
QUANTITY = "curvature"   # "curvature" (1/mm) or "radius" (mm) on the plots' y-axis
SHARE_Y = True           # same y-axis on every speed's plot so they can be compared

DIRECTION_NAMES = {"h": "Height (h)", "w": "Width (w)"}
# colours: first two slots of a palette that passes colour-vision checks
# (also different marker shapes, so the two groups never rely on colour alone)
STYLE = {"h": dict(color="#2a78d6", marker="o"), "w": dict(color="#eb6834", marker="s")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
# ---------------------------------------------------------------------------

NAME_RE = re.compile(
    r"^(?P<height>\d+(?:\.\d+)?)_(?P<speed>\d+(?:\.\d+)?)_(?P<count>\d+)_(?P<direction>[A-Za-z]+)"
    r"(?:_(?P<side>\d+))?$")


def parse_name(path):
    """Return (params dict, None) or (None, reason) for a file name."""
    stem = Path(path).stem
    m = NAME_RE.match(stem)
    if not m:
        return None, ("does not match totalheight_speed_printcount_side[_sidecount], "
                      "e.g. 2_80_1_h_1")
    d = m.group("direction").lower()
    if d not in DIRECTION_NAMES:
        return None, f"direction '{m.group('direction')}' is not 'h' or 'w'"
    num = lambda s: float(s) if "." in s else int(s)
    return dict(height_mm=num(m.group("height")), speed_mm_s=num(m.group("speed")),
                print_count=int(m.group("count")), direction=d,
                side_count=int(m.group("side")) if m.group("side") else None), None


def find_images(folder):
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in fq.EXTS and not p.name.startswith("."))
    ok, bad = [], []
    for p in files:
        params, why = parse_name(p)
        (ok if params else bad).append((p, params or why))
    return ok, bad


# ----------------------------------------------------------------------------
def run_fits(images, edge, scale, out_dir):
    rows = []
    for i, (path, params) in enumerate(images, 1):
        res = fq.analyze_image(path, edge, out_dir / "fits", scale=scale)
        row = {**params, "file": path.name, "edge": edge, **res}
        row["warnings"] = "; ".join(res.get("warnings", []))
        rows.append(row)
        status = res["error"] or (f"k = {res['k_mm']:.4g} 1/mm" if "k_mm" in res
                                  else f"k = {res['k_px']:.4g} 1/px")
        print(f"[{i}/{len(images)}] {path.name}: {status}"
              + (f"   WARNING: {row['warnings']}" if row["warnings"] else ""))
    return rows


def write_csv(rows, path, have_scale):
    cols = ["file", "height_mm", "speed_mm_s", "print_count", "direction", "side_count", "edge",
            "k_mm" if have_scale else "k_px", "R_mm" if have_scale else "R_px",
            "a", "b", "c", "a_mm", "c_mm", "mm_per_px", "resid_px", "n_points", "n_used",
            "warnings", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["speed_mm_s"], r["height_mm"], r["direction"],
                                             r["print_count"], r["side_count"] or 0)):
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})


# ----------------------------------------------------------------------------
def make_plot(rows, speed, quantity, have_scale, ymax, out_path):
    key = {"curvature": "k_mm" if have_scale else "k_px",
           "radius": "R_mm" if have_scale else "R_px"}[quantity]
    ylabel = {"curvature": f"Maximum curvature ({'1/mm' if have_scale else '1/px'})",
              "radius": f"Radius of curvature ({'mm' if have_scale else 'px'})"}[quantity]
    sel = [r for r in rows if r["speed_mm_s"] == speed and not r["error"] and r.get(key) is not None]

    heights = sorted({r["height_mm"] for r in sel})
    span = (max(heights) - min(heights)) if len(heights) > 1 else 1.0
    gap = min(np.diff(heights)) if len(heights) > 1 else 1.0
    dodge = {"h": -0.07 * gap, "w": +0.07 * gap}

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for d in ("h", "w"):
        st = STYLE[d]
        means_x, means_y = [], []
        for hgt in heights:
            vals = np.array([r[key] for r in sel if r["direction"] == d and r["height_mm"] == hgt])
            if vals.size == 0:
                continue
            xs = hgt + dodge[d] + np.linspace(-0.02, 0.02, vals.size) * gap * (vals.size > 1)
            ax.scatter(xs, vals, s=26, marker=st["marker"], color=st["color"], alpha=0.55,
                       linewidths=0, zorder=3)                       # individual measurements
            m = vals.mean()
            sd = vals.std(ddof=1) if vals.size > 1 else 0.0
            means_x.append(hgt + dodge[d]); means_y.append(m)
            if vals.size > 1:
                ax.errorbar(hgt + dodge[d], m, yerr=sd, color=st["color"], capsize=4, lw=1.4,
                            zorder=4, fmt="none")
            ax.scatter(hgt + dodge[d], m, s=80, marker=st["marker"], color=st["color"],
                       edgecolors=SURFACE, linewidths=1.5, zorder=5)  # mean
        if len(means_x) > 1:
            ax.plot(means_x, means_y, color=st["color"], lw=1.6, zorder=2)
        if means_x:   # only list directions that actually have data on this plot
            ax.scatter([], [], s=80, marker=st["marker"], color=st["color"],
                       label=DIRECTION_NAMES[d])

    ax.set_xlabel("Total print height (mm)", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_xticks(heights)
    ax.set_xticklabels([f"{h:g}" for h in heights])
    pad = 0.5 * gap if len(heights) > 1 else 1.0
    ax.set_xlim(min(heights) - pad, max(heights) + pad)
    ax.set_ylim(0, (ymax if ymax else max(r[key] for r in sel)) * 1.08)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    n = len(sel)
    ax.set_title(f"Print speed {speed:g} mm/s", loc="left", color=INK, fontsize=13,
                 fontweight="bold", pad=22)
    ax.text(0, 1.02, f"{n} measurement{'s' * (n != 1)}. Small markers: each measurement; "
            "large markers: mean (bars: ±1 SD where n ≥ 2)", transform=ax.transAxes,
            color=INK2, fontsize=8, va="bottom")
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def make_check_sheet(rows, edge, out_dir, path, cols=4, width=520):
    """Contact sheet of cropped fit overlays, so every fit can be checked by eye."""
    tiles = []
    for r in sorted(rows, key=lambda r: (r["speed_mm_s"], r["height_mm"], r["direction"],
                                         r["print_count"], r["side_count"] or 0)):
        if r["error"]:
            continue
        stem = Path(r["file"]).stem
        ov = cv2.imread(str(out_dir / "fits" / f"{stem}_fit_{edge}.png"))
        mk = cv2.imread(str(out_dir / "fits" / f"{stem}_mask.png"), 0)
        if ov is None or mk is None:
            continue
        ys, xs = np.nonzero(mk)
        pad = int(0.12 * max(np.ptp(xs), np.ptp(ys)))
        y0, y1 = max(ys.min() - pad, 0), ys.max() + pad
        x0, x1 = max(xs.min() - pad, 0), xs.max() + pad
        crop = ov[y0:y1, x0:x1]
        crop = cv2.resize(crop, (width, int(width * crop.shape[0] / crop.shape[1])),
                          interpolation=cv2.INTER_AREA)
        k = r.get("k_mm", r.get("k_px"))
        label = f"{stem}   k={k:.3g}" + ("  !" if r["warnings"] else "")
        cv2.rectangle(crop, (0, 0), (width, 26), (255, 255, 255), -1)
        cv2.putText(crop, label, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                    cv2.LINE_AA)
        tiles.append(crop)
    if not tiles:
        return
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                value=(255, 255, 255)) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), sheet)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Curvature vs print parameters for a folder of photos.")
    ap.add_argument("folder", nargs="?", default=FOLDER)
    ap.add_argument("--edge", choices=["top", "bottom"], default=EDGE)
    ap.add_argument("--scale", default=SCALE_FILE, help="path to scale.json")
    ap.add_argument("--quantity", choices=["curvature", "radius"], default=QUANTITY)
    args = ap.parse_args()

    folder = Path(args.folder.strip().strip("'\"")).expanduser()
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")
    out_dir = folder / OUTPUT_SUBDIR
    out_dir.mkdir(exist_ok=True)

    images, bad = find_images(folder)
    print(f"Found {len(images) + len(bad)} images: {len(images)} with a valid name, "
          f"{len(bad)} skipped.")
    for p, why in bad:
        print(f"  SKIPPED {p.name}: {why}")
    if not images:
        sys.exit("No correctly named images to analyse.")

    # ---- scale
    if args.scale:
        scale = json.load(open(Path(args.scale).expanduser()))
        scale["_path"] = args.scale
    else:
        scale = fq.find_scale(images[0][0])
    have_scale = scale is not None
    if have_scale:
        print(f"Scale: {scale['mm_per_px']:.6f} mm/px  (from {scale['_path']})")
    else:
        print("\n*** No scale.json found: results and plots will be in PIXELS. "
              "Run calibrate_scale.py first for mm. ***\n")

    # ---- fit every image
    rows = run_fits(images, args.edge, scale, out_dir)
    write_csv(rows, out_dir / "results.csv", have_scale)

    failed = [r for r in rows if r["error"]]
    warned = [r for r in rows if r["warnings"] and not r["error"]]

    # ---- plots (one per speed)
    key = {"curvature": "k_mm" if have_scale else "k_px",
           "radius": "R_mm" if have_scale else "R_px"}[args.quantity]
    ok = [r for r in rows if not r["error"]]
    speeds = sorted({r["speed_mm_s"] for r in ok})
    ymax = max((r[key] for r in ok), default=None) if SHARE_Y else None
    for sp in speeds:
        out = out_dir / f"plot_speed_{sp:g}.png"
        make_plot(rows, sp, args.quantity, have_scale, ymax, out)
        print(f"Saved plot: {out}")
    make_check_sheet(rows, args.edge, out_dir, out_dir / "fit_check_sheet.png")

    print(f"\nDone. {len(ok)} of {len(rows)} images fitted. Results: {out_dir / 'results.csv'}")
    print(f"Check every fit by eye in: {out_dir / 'fit_check_sheet.png'}")
    if failed:
        print("Failed:", ", ".join(f"{r['file']} ({r['error']})" for r in failed))
    if warned:
        print(f"{len(warned)} image(s) have scale warnings (see 'warnings' in results.csv): "
              + ", ".join(r["file"] for r in warned))


if __name__ == "__main__":
    main()