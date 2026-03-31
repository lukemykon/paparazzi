#!/usr/bin/env python3
"""
Standalone Lucas-Kanade optical flow visualiser.

Mirrors the Paparazzi luke_optical_flow module:
  - Shi-Tomasi corner detection with brightness mask (avoids white regions)
  - Lucas-Kanade tracking (cv2.calcOpticalFlowPyrLK)
  - Size-divergence computation (same as Paparazzi size_divergence.c)
  - VISUAL_DETECTION threshold check

Modes:
  full     — whole-frame divergence (original behaviour)
  center   — center 50% of frame only
  columns  — three vertical columns; scores shown in-image, best column highlighted

Usage:
    python3 optical_flow_viz.py                              # columns mode (default)
    python3 optical_flow_viz.py --mode full                  # original whole-frame
    python3 optical_flow_viz.py --mode center                # center crop only
    python3 optical_flow_viz.py --end 200                    # first 200 frames
    python3 optical_flow_viz.py --save-dir /tmp/of_out       # save frames to disk
    python3 optical_flow_viz.py --threshold 0.15             # custom threshold

Interactive controls:
    Space  — pause / resume
    q      — quit
    n      — step one frame (when paused)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Parameters matching luke_optical_flow / Paparazzi defaults ────────────────
MAX_CORNERS      = 50            # OPTICFLOW_MAX_TRACK_CORNERS
LK_WIN           = (21, 21)      # 2 * OPTICFLOW_WINDOW_SIZE + 1  (window_size=10)
LK_PYRAMID       = 1             # OPTICFLOW_PYRAMID_LEVEL
LK_CRITERIA      = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03)
DEFAULT_DIV_THRESHOLD = 0.2   # luke_of_divergence_threshold

# Mask out near-saturated pixels before corner detection so bright
# ceiling lights / white walls don't monopolise all tracked points.
BRIGHT_MASK_THRESHOLD = 220   # pixels above this value are excluded (0-255)

# Column split fractions [left_end, right_start] as fraction of frame width.
# Center column is wider to down-weight fisheye-distorted edges.
COLUMN_SPLITS   = (0.33, 0.67)  # left | center | right (centre boundaries)
COLUMN_OVERLAP  = 0.10          # each column extends this fraction into its neighbours
COLUMN_LABELS = ("LEFT", "CENTER", "RIGHT")

BORDER = 8   # px border width for obstacle indicator


# ── Frame loading ──────────────────────────────────────────────────────────────

def load_frames(img_dir: Path) -> list[Path]:
    """Return image paths sorted numerically by timestamp filename."""
    paths = sorted(img_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not paths:
        sys.exit(f"No .jpg files found in {img_dir}")
    return paths


# ── Optical flow ──────────────────────────────────────────────────────────────

def detect_corners(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray | None:
    """
    Detect corners using Shi-Tomasi (goodFeaturesToTrack).
    Uses a brightness mask to exclude near-saturated regions (white walls /
    ceiling lights) that otherwise dominate FAST detection.
    An additional spatial mask can restrict detection to a sub-region.
    Returns array of shape (N, 1, 2) or None if no corners found.
    """
    bright_mask = np.where(gray < BRIGHT_MASK_THRESHOLD, np.uint8(255), np.uint8(0))
    combined = cv2.bitwise_and(bright_mask, mask) if mask is not None else bright_mask

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=7,
        mask=combined,
        blockSize=7,
    )
    return pts  # shape (N,1,2) or None


def compute_flow(prev_gray: np.ndarray,
                 curr_gray: np.ndarray,
                 spatial_mask: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Lucas-Kanade tracking between two grayscale frames (optionally masked).
    Returns (pts_prev, pts_curr) with only successfully tracked points.
    """
    pts_prev = detect_corners(prev_gray, spatial_mask)
    if pts_prev is None or len(pts_prev) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts_prev,
        None,
        winSize=LK_WIN,
        maxLevel=LK_PYRAMID,
        criteria=LK_CRITERIA,
    )

    ok = status.flatten() == 1
    return pts_prev[ok].reshape(-1, 2), pts_curr[ok].reshape(-1, 2)


# ── Divergence (mirrors size_divergence.c) ────────────────────────────────────

def compute_divergence(pts_prev: np.ndarray, pts_curr: np.ndarray) -> float:
    """
    Size-divergence: for every pair of tracked points, compute the relative
    change in inter-point distance.  Return the median across all pairs.
    """
    n = len(pts_prev)
    if n < 2:
        return 0.0

    divs = []
    for i in range(n):
        for j in range(i + 1, n):
            d_prev = np.linalg.norm(pts_prev[i] - pts_prev[j])
            d_curr = np.linalg.norm(pts_curr[i] - pts_curr[j])
            if d_prev > 1e-6:
                divs.append((d_curr - d_prev) / d_prev)

    return float(np.median(divs)) if divs else 0.0


# ── Column helpers ─────────────────────────────────────────────────────────────

def column_x_bounds(w: int) -> list[tuple[int, int]]:
    """Return (x_start, x_end) pixel bounds for each column, with overlap."""
    x1  = int(w * COLUMN_SPLITS[0])
    x2  = int(w * COLUMN_SPLITS[1])
    ovl = int(w * COLUMN_OVERLAP)
    return [
        (0,           min(x1 + ovl, w)),   # left:   starts at 0, bleeds right
        (max(x1 - ovl, 0), min(x2 + ovl, w)),  # center: bleeds both sides
        (max(x2 - ovl, 0), w),             # right:  bleeds left, ends at w
    ]


def make_column_mask(h: int, w: int, x_start: int, x_end: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, x_start:x_end] = 255
    return mask


def compute_column_scores(prev_gray: np.ndarray,
                          curr_gray: np.ndarray
                          ) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """
    For each of the three vertical columns, run LK flow and compute divergence.
    Returns list of (divergence, pts_prev, pts_curr) per column.
    """
    h, w = prev_gray.shape
    results = []
    for x_start, x_end in column_x_bounds(w):
        mask = make_column_mask(h, w, x_start, x_end)
        pp, pc = compute_flow(prev_gray, curr_gray, spatial_mask=mask)
        div = compute_divergence(pp, pc)
        results.append((div, pp, pc))
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

# Colour ramp: low divergence → green, high → red
def _divergence_colour(div: float, threshold: float) -> tuple[int, int, int]:
    """BGR colour from green (0) to red (threshold+)."""
    t = min(abs(div) / max(threshold, 1e-6), 1.0)
    r = int(255 * t)
    g = int(255 * (1.0 - t))
    return (0, g, r)


def draw_visualization_full(frame: np.ndarray,
                             pts_prev: np.ndarray,
                             pts_curr: np.ndarray,
                             divergence: float,
                             threshold: float,
                             frame_idx: int) -> np.ndarray:
    """Original whole-frame visualisation."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    obstacle = divergence > threshold
    border_colour = (0, 0, 220) if obstacle else (0, 200, 0)
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), border_colour, BORDER)

    for p_prev, p_curr in zip(pts_prev.astype(int), pts_curr.astype(int)):
        cv2.circle(vis, tuple(p_prev), 4, (0, 255, 0), -1)
        cv2.arrowedLine(vis, tuple(p_prev), tuple(p_curr),
                        (255, 100, 0), 1, tipLength=0.4)

    lines = [
        f"Frame {frame_idx}",
        f"Tracked: {len(pts_prev)}",
        f"Divergence: {divergence:+.4f}",
        f"Threshold:  {threshold:.3f}",
        "OBSTACLE" if obstacle else "clear",
    ]
    colours = [(220, 220, 220)] * 4 + [(0, 0, 220) if obstacle else (0, 200, 0)]
    y0 = 22
    for text, colour in zip(lines, colours):
        cv2.putText(vis, text, (BORDER + 4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (BORDER + 4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y0 += 20
    return vis


def draw_visualization_columns(frame: np.ndarray,
                                col_results: list[tuple[float, np.ndarray, np.ndarray]],
                                threshold: float,
                                frame_idx: int) -> np.ndarray:
    """
    Three-column visualisation:
      - Vertical dividers between columns
      - Per-column divergence score and tracked-point count
      - Colour-coded column tint (green=low flow, red=high flow)
      - Bold highlight on the best (lowest divergence) column
      - Flow arrows drawn per column
    """
    vis = frame.copy()
    h, w = vis.shape[:2]
    bounds = column_x_bounds(w)

    divs   = [r[0] for r in col_results]
    best   = int(np.argmin(divs))

    # ── Per-column tinted overlay + flow vectors ──────────────────────────────
    for idx, ((x0, x1), (div, pp, pc)) in enumerate(zip(bounds, col_results)):
        colour = _divergence_colour(div, threshold)

        # Semi-transparent tint
        overlay = vis.copy()
        cv2.rectangle(overlay, (x0, 0), (x1 - 1, h - 1), colour, -1)
        alpha = 0.18 if idx != best else 0.10   # best column stays cleaner
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # Flow arrows
        for p_prev, p_curr in zip(pp.astype(int), pc.astype(int)):
            cv2.circle(vis, tuple(p_prev), 3, (0, 255, 0), -1)
            cv2.arrowedLine(vis, tuple(p_prev), tuple(p_curr),
                            (255, 120, 0), 1, tipLength=0.4)

        # Column border — thicker & brighter for best column
        border_w  = 4 if idx == best else 2
        bdr_colour = (0, 255, 80) if idx == best else (160, 160, 160)
        cv2.rectangle(vis, (x0, 0), (x1 - 1, h - 1), bdr_colour, border_w)

        # ── Score box centred in column ───────────────────────────────────────
        cx = (x0 + x1) // 2
        label       = COLUMN_LABELS[idx]
        score_text  = f"div: {div:+.3f}"
        track_text  = f"pts: {len(pp)}"
        best_text   = "< BEST" if idx == best else ""

        # Background pill
        box_w, box_h = 110, 68
        bx0 = cx - box_w // 2
        by0 = h // 2 - box_h // 2
        overlay2 = vis.copy()
        cv2.rectangle(overlay2, (bx0, by0), (bx0 + box_w, by0 + box_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.55, vis, 0.45, 0, vis)

        # Text lines inside box
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        ty    = by0 + 16
        for text, col in [
            (label,      (220, 220, 220)),
            (score_text, colour),
            (track_text, (180, 180, 180)),
            (best_text,  (0, 255, 80)),
        ]:
            cv2.putText(vis, text, (bx0 + 6, ty), font, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, text, (bx0 + 6, ty), font, scale, col,       1, cv2.LINE_AA)
            ty += 16

    # ── Top-left frame info ────────────────────────────────────────────────────
    best_label = COLUMN_LABELS[best]
    info_lines = [
        f"Frame {frame_idx}",
        f"Go: {best_label}",
        f"Thr: {threshold:.3f}",
    ]
    y0 = 18
    for text in info_lines:
        cv2.putText(vis, text, (BORDER + 4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (BORDER + 4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        y0 += 17

    return vis


def draw_visualization_center(frame: np.ndarray,
                               pts_prev: np.ndarray,
                               pts_curr: np.ndarray,
                               divergence: float,
                               threshold: float,
                               frame_idx: int) -> np.ndarray:
    """Center-crop mode: same as full but faded sides show the crop region."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    x0 = int(w * 0.25)
    x1 = int(w * 0.75)

    # Darken the unused sides
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (x0, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)

    # Center crop border
    obstacle = divergence > threshold
    bdr = (0, 0, 220) if obstacle else (0, 200, 0)
    cv2.rectangle(vis, (x0, 0), (x1 - 1, h - 1), bdr, 3)

    for p_prev, p_curr in zip(pts_prev.astype(int), pts_curr.astype(int)):
        cv2.circle(vis, tuple(p_prev), 4, (0, 255, 0), -1)
        cv2.arrowedLine(vis, tuple(p_prev), tuple(p_curr),
                        (255, 100, 0), 1, tipLength=0.4)

    lines = [
        f"Frame {frame_idx}",
        f"Tracked: {len(pts_prev)}",
        f"Div: {divergence:+.4f}",
        f"Thr: {threshold:.3f}",
        "OBSTACLE" if obstacle else "clear",
    ]
    cols = [(220, 220, 220)] * 4 + [(0, 0, 220) if obstacle else (0, 200, 0)]
    y0 = 22
    for text, colour in zip(lines, cols):
        cv2.putText(vis, text, (BORDER + 4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (BORDER + 4, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y0 += 20
    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-dir", default=str(Path.home() / "tud/Q3_MAV/training_img"),
                   help="Directory with .jpg frames")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=-1, help="-1 = all frames")
    p.add_argument("--threshold", type=float, default=DEFAULT_DIV_THRESHOLD)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mode", choices=["full", "center", "columns"], default="columns",
                   help="Visualisation mode (default: columns)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    img_dir = Path(args.img_dir).expanduser()
    paths   = load_frames(img_dir)

    end   = len(paths) if args.end < 0 else min(args.end, len(paths))
    paths = paths[args.start:end]
    print(f"Mode: {args.mode}  |  frames {args.start}–{end - 1} ({len(paths)} frames)")
    print(f"Divergence threshold: {args.threshold}")

    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving to: {save_dir}")
    else:
        cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Optical Flow", 520, 480)

    delay_ms = max(1, 1000 // args.fps)
    paused   = False
    prev_gray = None

    # Center-crop mask (reused each frame once shape is known)
    center_mask = None

    for i, path in enumerate(paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Warning: could not read {path.name}, skipping")
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = curr_gray.shape

        if prev_gray is not None:
            if args.mode == "full":
                pp, pc = compute_flow(prev_gray, curr_gray)
                div    = compute_divergence(pp, pc)
                vis    = draw_visualization_full(frame, pp, pc, div, args.threshold,
                                                 args.start + i)
                print(f"  [{args.start + i:5d}] tracked={len(pp):2d}  "
                      f"div={div:+.4f}  "
                      f"{'OBSTACLE' if div > args.threshold else 'clear'}")

            elif args.mode == "center":
                if center_mask is None:
                    center_mask = make_column_mask(h, w, int(w * 0.25), int(w * 0.75))
                pp, pc = compute_flow(prev_gray, curr_gray, center_mask)
                div    = compute_divergence(pp, pc)
                vis    = draw_visualization_center(frame, pp, pc, div, args.threshold,
                                                   args.start + i)
                print(f"  [{args.start + i:5d}] tracked={len(pp):2d}  "
                      f"div={div:+.4f}  "
                      f"{'OBSTACLE' if div > args.threshold else 'clear'}")

            else:  # columns
                col_results = compute_column_scores(prev_gray, curr_gray)
                divs  = [r[0] for r in col_results]
                best  = int(np.argmin(divs))
                vis   = draw_visualization_columns(frame, col_results, args.threshold,
                                                   args.start + i)
                print(f"  [{args.start + i:5d}]  "
                      f"L={divs[0]:+.3f}  C={divs[1]:+.3f}  R={divs[2]:+.3f}  "
                      f"→ {COLUMN_LABELS[best]}")

        else:
            vis = frame.copy()
            cv2.putText(vis, "First frame (no flow yet)", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if save_dir:
            out_path = save_dir / f"of_{args.start + i:05d}_{path.stem}.jpg"
            cv2.imwrite(str(out_path), vis)
        else:
            cv2.imshow("Optical Flow", vis)
            while True:
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
                if key == ord('q'):
                    print("Quit.")
                    cv2.destroyAllWindows()
                    return
                elif key == ord(' '):
                    paused = not paused
                    if not paused:
                        break
                elif key == ord('n') and paused:
                    break
                elif not paused:
                    break

        prev_gray = curr_gray

    if not save_dir:
        cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
