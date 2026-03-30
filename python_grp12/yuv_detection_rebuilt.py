#!/usr/bin/env python3
"""
Interactive YCrCb mask tuner.

What this version changes:
- Accurate channel names: Y, Cr, Cb
- Dedicated control window so sliders are always visible
- Clear slider labels: Y_min, Y_max, Cr_min, Cr_max, Cb_min, Cb_max
- Automatic 90 degree image rotation
- Larger, resizable preview/control windows
- Side-by-side preview with a readable live values panel

Usage:
    python3 yuv_detection_rebuilt.py --image path/to/image.jpg
    python3 yuv_detection_rebuilt.py --image path/to/image.jpg --preset skin
    python3 yuv_detection_rebuilt.py --image path/to/image.jpg --save overlay.png

Keys:
    q / ESC   quit
    s         save current overlay (and mask) when --save is provided
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


# Initial presets in OpenCV's YCrCb order: [Y, Cr, Cb]
PRESETS = {
    "skin":   (np.array([0, 133, 80]),  np.array([255, 173, 120])),
    "green":  (np.array([0, 0, 60]),    np.array([255, 90, 110])),
    "red":    (np.array([0, 150, 100]), np.array([255, 200, 130])),
    "blue":   (np.array([0, 0, 80]),    np.array([255, 100, 140])),
    "yellow": (np.array([80, 110, 60]), np.array([255, 170, 110])),
    "white":  (np.array([200, 100, 100]), np.array([255, 130, 130])),
    "black":  (np.array([0, 90, 90]),   np.array([60, 130, 130])),
}

ROTATE_MODE = cv2.ROTATE_90_COUNTERCLOCKWISE
PREVIEW_WINDOW = "YCrCb Mask Preview"
CONTROL_WINDOW = "YCrCb Controls"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive YCrCb Color Mask Tool")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--preset",
        default="skin",
        choices=sorted(PRESETS.keys()),
        help="Initial slider preset",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Base path for saving current overlay and mask when you press 's'",
    )
    return parser.parse_args()


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"[ERROR] Cannot open image: '{path}'")
    return cv2.rotate(img, ROTATE_MODE)


def bgr_to_ycrcb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)


def build_mask(img_ycrcb: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(img_ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def apply_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grey_bgr = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

    highlight = np.full_like(img_bgr, (255, 255, 0), dtype=np.uint8)  # cyan in BGR
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, highlight, alpha, 0)

    result = grey_bgr.copy()
    matched = mask > 0
    result[matched] = blended[matched]
    return result


def resize_keep_aspect(img: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_title_bar(img: np.ndarray, title: str) -> np.ndarray:
    bar_h = 38
    panel = cv2.copyMakeBorder(img, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(panel, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def make_mask_panel(mask: np.ndarray, height: int) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    if mask_bgr.shape[0] != height:
        scale = height / mask_bgr.shape[0]
        new_w = max(1, int(round(mask_bgr.shape[1] * scale)))
        mask_bgr = cv2.resize(mask_bgr, (new_w, height), interpolation=cv2.INTER_NEAREST)
    return mask_bgr


def make_info_panel(bounds: dict[str, int], masked_px: int, total_px: int, height: int) -> np.ndarray:
    panel_w = 340
    panel = np.full((height, panel_w, 3), 32, dtype=np.uint8)

    lines = [
        "Live thresholds",
        f"Y_min  : {bounds['Y_min']:3d}",
        f"Y_max  : {bounds['Y_max']:3d}",
        f"Cr_min : {bounds['Cr_min']:3d}",
        f"Cr_max : {bounds['Cr_max']:3d}",
        f"Cb_min : {bounds['Cb_min']:3d}",
        f"Cb_max : {bounds['Cb_max']:3d}",
        "",
        f"Masked  : {masked_px:,}",
        f"Total   : {total_px:,}",
        f"Percent : {100.0 * masked_px / max(total_px, 1):5.1f}%",
        "",
        "Keys",
        "q / ESC : quit",
        "s       : save current result",
    ]

    y = 42
    for i, text in enumerate(lines):
        if text == "":
            y += 16
            continue
        font_scale = 0.92 if i == 0 else 0.72
        thickness = 2 if i == 0 else 1
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)
        y += 34 if i == 0 else 28

    return panel


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    if h > height:
        return resize_keep_aspect(img, w, height)
    top = (height - h) // 2
    bottom = height - h - top
    return cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))


def create_trackbar(name: str, window: str, value: int) -> None:
    cv2.createTrackbar(name, window, int(value), 255, lambda _x: None)


def set_initial_trackbars(window: str, lower: np.ndarray, upper: np.ndarray) -> None:
    create_trackbar("Y_min", window, int(lower[0]))
    create_trackbar("Y_max", window, int(upper[0]))
    create_trackbar("Cr_min", window, int(lower[1]))
    create_trackbar("Cr_max", window, int(upper[1]))
    create_trackbar("Cb_min", window, int(lower[2]))
    create_trackbar("Cb_max", window, int(upper[2]))


def read_bounds(window: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    vals = {
        "Y_min": cv2.getTrackbarPos("Y_min", window),
        "Y_max": cv2.getTrackbarPos("Y_max", window),
        "Cr_min": cv2.getTrackbarPos("Cr_min", window),
        "Cr_max": cv2.getTrackbarPos("Cr_max", window),
        "Cb_min": cv2.getTrackbarPos("Cb_min", window),
        "Cb_max": cv2.getTrackbarPos("Cb_max", window),
    }

    # Keep min/max pairs valid.
    if vals["Y_min"] > vals["Y_max"]:
        vals["Y_max"] = vals["Y_min"]
        cv2.setTrackbarPos("Y_max", window, vals["Y_max"])
    if vals["Cr_min"] > vals["Cr_max"]:
        vals["Cr_max"] = vals["Cr_min"]
        cv2.setTrackbarPos("Cr_max", window, vals["Cr_max"])
    if vals["Cb_min"] > vals["Cb_max"]:
        vals["Cb_max"] = vals["Cb_min"]
        cv2.setTrackbarPos("Cb_max", window, vals["Cb_max"])

    lower = np.array([vals["Y_min"], vals["Cr_min"], vals["Cb_min"]], dtype=np.uint8)
    upper = np.array([vals["Y_max"], vals["Cr_max"], vals["Cb_max"]], dtype=np.uint8)
    return lower, upper, vals


def save_outputs(base_path: str, overlay: np.ndarray, mask: np.ndarray) -> None:
    base = Path(base_path)
    if base.suffix:
        overlay_path = base
        mask_path = base.with_name(f"{base.stem}_mask.png")
    else:
        overlay_path = base.parent / f"{base.name}_overlay.png"
        mask_path = base.parent / f"{base.name}_mask.png"

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(mask_path), mask)
    print(f"[INFO] Saved overlay: {overlay_path}")
    print(f"[INFO] Saved mask   : {mask_path}")


def build_preview(original: np.ndarray, overlay: np.ndarray, mask: np.ndarray, bounds: dict[str, int]) -> np.ndarray:
    # Large but controlled sizing so the window stays practical.
    original_small = resize_keep_aspect(original, 520, 720)
    overlay_small = resize_keep_aspect(overlay, 520, 720)

    common_h = max(original_small.shape[0], overlay_small.shape[0])
    original_small = pad_to_height(original_small, common_h)
    overlay_small = pad_to_height(overlay_small, common_h)
    mask_panel = make_mask_panel(mask, common_h)

    total_px = mask.size
    masked_px = int(np.count_nonzero(mask))
    info_panel = make_info_panel(bounds, masked_px, total_px, common_h)

    original_small = add_title_bar(original_small, "Original (rotated 90 deg)")
    overlay_small = add_title_bar(overlay_small, "Overlay")
    mask_panel = add_title_bar(mask_panel, "Mask")
    info_panel = add_title_bar(info_panel, "Info")

    max_h = max(original_small.shape[0], overlay_small.shape[0], mask_panel.shape[0], info_panel.shape[0])
    original_small = pad_to_height(original_small, max_h)
    overlay_small = pad_to_height(overlay_small, max_h)
    mask_panel = pad_to_height(mask_panel, max_h)
    info_panel = pad_to_height(info_panel, max_h)

    spacer = np.full((max_h, 14, 3), 18, dtype=np.uint8)
    preview = np.hstack([original_small, spacer, overlay_small, spacer, mask_panel, spacer, info_panel])
    return preview


def main() -> None:
    args = parse_args()

    img_bgr = load_image(args.image)
    img_ycrcb = bgr_to_ycrcb(img_bgr)

    lower0, upper0 = PRESETS[args.preset]

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, 1700, 900)
    cv2.resizeWindow(CONTROL_WINDOW, 760, 320)

    set_initial_trackbars(CONTROL_WINDOW, lower0, upper0)

    print("[INFO] Controls are in a separate window so sliders remain visible.")
    print("[INFO] Press 'q' or ESC to quit.")
    if args.save:
        print("[INFO] Press 's' to save the current overlay and mask.")

    while True:
        lower, upper, bounds = read_bounds(CONTROL_WINDOW)
        mask = build_mask(img_ycrcb, lower, upper)
        overlay = apply_overlay(img_bgr, mask)
        preview = build_preview(img_bgr, overlay, mask, bounds)

        cv2.imshow(PREVIEW_WINDOW, preview)

        # Keep a small visible area in the controls window even though it mainly hosts sliders.
        controls_img = np.full((130, 740, 3), 36, dtype=np.uint8)
        summary = (
            f"Y[{bounds['Y_min']},{bounds['Y_max']}]   "
            f"Cr[{bounds['Cr_min']},{bounds['Cr_max']}]   "
            f"Cb[{bounds['Cb_min']},{bounds['Cb_max']}]"
        )
        cv2.putText(controls_img, "Move sliders below", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(controls_img, summary, (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow(CONTROL_WINDOW, controls_img)

        key = cv2.waitKey(25) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s") and args.save:
            save_outputs(args.save, overlay, mask)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
