import cv2
import numpy as np
import os
import datetime

# ==============================
# PARAMETERS (TUNE THESE)
# ==============================

MIN_CONTOUR_AREA  = 500    # Ignore blobs smaller than this (noise filter)
MIN_TEXTURE_SCORE = 15.0   # Plants have high edge density; grass is smooth
MAX_SOLIDITY      = 0.85   # Plants are jagged; solid blobs (>0.85) are likely grass
MIN_COMPLEXITY    = 1.5    # Plants have complex outlines; grass blobs are simple
MIN_COLOR_STDDEV  = 10.0   # Plants have varied colours; grass is uniform

BOTTOM_CROP_RATIO = 0.40   # Crop the bottom 40% of each image (mostly ground)
BOX_MERGE_DISTANCE = 15    # Merge final boxes within this many pixels
MIN_BOX_AREA       = 2000  # Remove final boxes smaller than this (width × height)

# If the straight zone has at most this much MORE plant coverage than the
# best turning direction, still prefer straight (straight-bias margin).
# E.g. 0.10 means: "go straight unless turning saves >10 % of total plant area"
STRAIGHT_BIAS = 0.10

LOWER_GREEN = np.array([28,  20,  50])   # HSV lower bound for green
UPPER_GREEN = np.array([85, 255, 255])   # HSV upper bound for green


# Use current working directory for output_results
INPUT_PATH = os.path.expanduser("~/paparazzi/python_grp12/raul_img.jpg")
BASE_OUTPUT_PATH = os.path.expanduser("~/paparazzi/python_grp12/output_results")


# main output path with timestamp

OUTPUT_PATH = os.path.join(
    BASE_OUTPUT_PATH,
    f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# subfolders for intermediate steps
INTERMEDIATE_FOLDERS = {
    'input': os.path.join(OUTPUT_PATH, '01_input'),
    'prepared': os.path.join(OUTPUT_PATH, '02_prepared'),
    'green_mask': os.path.join(OUTPUT_PATH, '03_green_mask'),
    'texture_map': os.path.join(OUTPUT_PATH, '04_texture_map'),
    'plant_mask': os.path.join(OUTPUT_PATH, '05_plant_mask'),
    'canny_edges': os.path.join(OUTPUT_PATH, '06_canny_edges'),
    'final': os.path.join(OUTPUT_PATH, '99_final'),
    'montage': os.path.join(OUTPUT_PATH, '98_montage'),
}
for folder in INTERMEDIATE_FOLDERS.values():
    os.makedirs(folder, exist_ok=True)



# Montage function to combine intermediate images
def create_montage(prepared, green_mask, canny_edges, final_overlay, filename):
    # Convert green_mask and canny_edges to 3-channel for stacking
    green_mask_color = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    canny_edges_color = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)

    # Resize all to the same size (use prepared image size)
    h, w = prepared.shape[:2]
    def resize(img):
        return cv2.resize(img, (w, h))
    prepared = resize(prepared)
    green_mask_color = resize(green_mask_color)
    canny_edges_color = resize(canny_edges_color)
    final_overlay = resize(final_overlay)

    # Add labels to each image
    def add_label(img, text):
        labeled = img.copy()
        cv2.rectangle(labeled, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(labeled, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return labeled
    prepared = add_label(prepared, 'Original (rotated/cropped)')
    green_mask_color = add_label(green_mask_color, 'Green Mask')
    canny_edges_color = add_label(canny_edges_color, 'Canny Edges')
    final_overlay = add_label(final_overlay, 'Final Bounding Box')

    # Stack horizontally
    montage = np.hstack([prepared, green_mask_color, canny_edges_color, final_overlay])
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['montage'], filename), montage)

# New function: create a montage with just green mask and bounding box image (no arrow)
def create_simple_montage(green_mask, bbox_img, filename):
    h, w = bbox_img.shape[:2]
    green_mask_color = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    green_mask_color = cv2.resize(green_mask_color, (w, h))
    bbox_img = cv2.resize(bbox_img, (w, h))
    # Add labels
    def add_label(img, text):
        labeled = img.copy()
        cv2.rectangle(labeled, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(labeled, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        return labeled
    green_mask_color = add_label(green_mask_color, 'Green Mask')
    bbox_img = add_label(bbox_img, 'Bounding Box')
    montage = np.hstack([green_mask_color, bbox_img])
    out_path = os.path.join(INTERMEDIATE_FOLDERS['montage'], f"simple_{filename}")
    cv2.imwrite(out_path, montage)
    return out_path


# ==============================
# IMAGE PRE-PROCESSING
# ==============================

def prepare_image(image):
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h = image.shape[0]
    return image[: int(h * (1.0 - BOTTOM_CROP_RATIO))]


# ==============================
# GREEN MASK
# ==============================

def extract_green(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab = cv2.merge((cv2.equalizeHist(l), a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    mask = cv2.inRange(cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV),
                       LOWER_GREEN, UPPER_GREEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask


# ==============================
# PLANT vs GRASS CLASSIFIER
# ==============================

def build_texture_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.uint8(np.clip(np.sqrt(gx**2 + gy**2), 0, 255))
    return cv2.GaussianBlur(mag, (31, 31), 0)


def is_plant(contour, texture_map, green_mask, image):
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return False
    mask = np.zeros(green_mask.shape, np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.bitwise_and(mask, green_mask)
    px_tex   = texture_map[mask > 0]
    px_color = image[mask > 0]
    texture   = float(np.mean(px_tex))  if len(px_tex)   > 0 else 0.0
    color_std = float(np.std(px_color)) if len(px_color) > 0 else 0.0
    area      = cv2.contourArea(contour)
    perim     = cv2.arcLength(contour, True)
    complexity = (perim ** 2) / (4 * np.pi * area) if area > 0 else 0.0
    solidity   = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
    return (
        texture   >= MIN_TEXTURE_SCORE and
        color_std >= MIN_COLOR_STDDEV  and
        (complexity >= MIN_COMPLEXITY or solidity <= MAX_SOLIDITY)
    )


# ==============================
# BOX FILTERING & MERGING
# ==============================

def remove_wide_boxes(boxes):
    return [(x, y, w, h) for (x, y, w, h) in boxes if w <= h * 2]

def remove_small_boxes(boxes):
    return [(x, y, w, h) for (x, y, w, h) in boxes if w * h >= MIN_BOX_AREA]

def merge_boxes(boxes, distance):
    if not boxes:
        return []
    boxes   = [list(b) for b in boxes]
    changed = True
    while changed:
        changed, merged, used = False, [], [False] * len(boxes)
        for i, (x, y, w, h) in enumerate(boxes):
            if used[i]:
                continue
            x2, y2, used[i] = x + w, y + h, True
            again = True
            while again:
                again = False
                for j, (bx, by, bw, bh) in enumerate(boxes):
                    if used[j]:
                        continue
                    bx2, by2 = bx + bw, by + bh
                    if (max(0, max(bx - x2, x - bx2)) <= distance and
                            max(0, max(by - y2, y - by2)) <= distance):
                        x, y   = min(x, bx), min(y, by)
                        x2, y2 = max(x2, bx2), max(y2, by2)
                        used[j], again, changed = True, True, True
            merged.append([x, y, x2 - x, y2 - y])
        boxes = merged
    return boxes


# ==============================
# DIRECTION CONFIDENCE
# ==============================

def compute_direction_confidence(boxes, img_width):
    """
    Divide the image into 3 equal vertical zones: Left | Straight | Right.

    Scores represent PLANT LOAD per zone — how much vegetation the drone
    would encounter flying in that direction.  The drone should avoid plants,
    so it heads toward the zone with the LEAST plant coverage.

    Decision rule (obstacle-avoidance, straight-preferred):
      1. Compute plant-load % per zone (higher = more plants = worse).
      2. STRAIGHT wins by default.
      3. A turn is chosen only when:
             straight_load  >  best_turn_load + STRAIGHT_BIAS * total_load
         i.e. turning must reduce plant exposure by more than STRAIGHT_BIAS
         of total plant area before we bother changing direction.
      4. When a turn is warranted, pick the side with LESS plant coverage.

    Returns
    -------
    zone_scores  : dict  raw pixel-area totals per zone
    confidences  : dict  plant-load % per zone (higher = more plants)
    direction    : str   recommended direction ('LEFT' | 'STRAIGHT' | 'RIGHT')
    per_box      : list  per-box zone breakdown for annotation
    """
    zone_w = img_width / 3.0
    zone_scores = {'left': 0.0, 'straight': 0.0, 'right': 0.0}
    per_box = []

    for (x, y, w, h) in boxes:
        box_area = w * h
        if box_area == 0:
            continue
        x_start, x_end = x, x + w
        zone_bounds = {
            'left':     (0,              int(zone_w)),
            'straight': (int(zone_w),    int(2 * zone_w)),
            'right':    (int(2 * zone_w), img_width),
        }
        box_zone = {}
        for name, (z0, z1) in zone_bounds.items():
            overlap      = max(0, min(x_end, z1) - max(x_start, z0))
            fraction     = overlap / w
            contribution = fraction * box_area
            zone_scores[name] += contribution
            box_zone[name]     = round(fraction * 100, 1)

        dom = max(box_zone, key=box_zone.get)
        per_box.append({'box': (x, y, w, h), 'zones': box_zone, 'dominant': dom})

    total = sum(zone_scores.values())
    if total == 0:
        confidences = {'left': 0.0, 'straight': 0.0, 'right': 0.0}
        direction   = 'STRAIGHT'
        return zone_scores, confidences, direction, per_box

    confidences = {k: round(v / total * 100, 1) for k, v in zone_scores.items()}

    straight_load = zone_scores['straight']
    # Best turn = whichever side has LESS plant load
    best_turn_load = min(zone_scores['left'], zone_scores['right'])

    # Only turn if doing so avoids meaningfully more plants than going straight
    if straight_load <= best_turn_load + STRAIGHT_BIAS * total:
        direction = 'STRAIGHT'
    else:
        direction = 'RIGHT' if zone_scores['right'] <= zone_scores['left'] else 'LEFT'

    return zone_scores, confidences, direction, per_box


# ==============================
# DRAW OVERLAY
# ==============================

ZONE_COLORS = {
    'left':     (255, 80,  80),
    'straight': (80,  200, 80),
    'right':    (80,  80,  255),
}
ARROW_COLOR = (0, 255, 255)
BOX_COLOR   = (0, 255, 0)


def draw_zone_dividers(overlay, img_width, img_height):
    zone_w = img_width // 3
    for x in [zone_w, 2 * zone_w]:
        for y in range(0, img_height, 20):
            cv2.line(overlay, (x, y), (x, min(y + 12, img_height)),
                     (220, 220, 220), 2)


def draw_zone_labels(overlay, img_width, img_height, confidences, direction):
    """Show plant-load % per zone; highlight the chosen direction."""
    zone_w  = img_width // 3
    centres = [zone_w // 2, zone_w + zone_w // 2, 2 * zone_w + zone_w // 2]
    names   = ['LEFT', 'STRAIGHT', 'RIGHT']
    keys    = ['left', 'straight', 'right']

    for cx, name, key in zip(centres, names, keys):
        load     = confidences[key]
        text     = f"{name}  {load:.0f}%"
        is_chosen = (name == direction)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx, ty = cx - tw // 2, 38
        bg_color = ZONE_COLORS[key] if is_chosen else (30, 30, 30)
        fg_color = (0, 0, 0)       if is_chosen else ZONE_COLORS[key]
        cv2.rectangle(overlay, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                      bg_color, -1)
        cv2.rectangle(overlay, (tx - 6, ty - th - 6), (tx + tw + 6, ty + 6),
                      ZONE_COLORS[key], 2)
        cv2.putText(overlay, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, fg_color, 2)


def draw_direction_arrow(overlay, img_width, img_height, direction):
    cx, cy, length = img_width // 2, img_height - 30, 60
    if direction == 'LEFT':
        pt1, pt2 = (cx + length, cy), (cx - length, cy)
    elif direction == 'RIGHT':
        pt1, pt2 = (cx - length, cy), (cx + length, cy)
    else:
        pt1, pt2 = (cx, cy + length), (cx, cy - length)
    cv2.arrowedLine(overlay, pt1, pt2, ARROW_COLOR, 4, tipLength=0.35)
    cv2.putText(overlay, f"GO {direction}", (cx - 55, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ARROW_COLOR, 2)


def draw_results(image, confidences, direction, per_box):
    overlay = image.copy()
    # Only draw bounding boxes and their labels
    for info in per_box:
        x, y, bw, bh = info['box']
        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), BOX_COLOR, 3)
        label = f"PLANT [{info['dominant'].upper()}]"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (x, y - lh - 10), (x + lw + 4, y), (0, 0, 0), -1)
        cv2.putText(overlay, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
    return overlay


# ==============================
# MAIN PIPELINE
# ==============================



def process_image(image, filename):
    h_img, w_img = image.shape[:2]

    # Save input image
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['input'], filename), image)

    # Step 1: Prepare image
    prepared = prepare_image(image)
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['prepared'], filename), prepared)

    # Step 2: Green mask
    green_mask = extract_green(prepared)
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['green_mask'], filename), green_mask)

    # Step 3: Texture map
    texture_map = build_texture_map(prepared)
    # Normalize for visualization
    texture_map_vis = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['texture_map'], filename), texture_map_vis)

    # Step 4: Canny edge detection (on prepared image)
    canny_edges = cv2.Canny(prepared, 100, 200)
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['canny_edges'], filename), canny_edges)

    # Step 5: Plant mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plant_mask = np.zeros(green_mask.shape, np.uint8)
    for cnt in contours:
        if is_plant(cnt, texture_map, green_mask, prepared):
            cv2.drawContours(plant_mask, [cnt], -1, 255, -1)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['plant_mask'], filename), plant_mask)

    # Step 6: Final results
    contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
    raw_boxes   = remove_wide_boxes(raw_boxes)
    raw_boxes   = remove_small_boxes(raw_boxes)
    final_boxes = merge_boxes(raw_boxes, BOX_MERGE_DISTANCE)

    _, confidences, direction, per_box = compute_direction_confidence(final_boxes, w_img)


    # Bounding box image (no arrow):
    bbox_img = draw_results(prepared, confidences, direction, per_box)
    cv2.imwrite(os.path.join(INTERMEDIATE_FOLDERS['final'], filename), bbox_img)

    # Create montage image for visual summary (full)
    create_montage(prepared, green_mask, canny_edges, bbox_img, filename)

    # Create simple montage: green mask + bounding box (no arrow)
    create_simple_montage(green_mask, bbox_img, filename)

    print(f"  Plants detected : {len(final_boxes)}")
    print(f"  Plant load      : LEFT {confidences['left']:.1f}%  "
          f"STRAIGHT {confidences['straight']:.1f}%  "
          f"RIGHT {confidences['right']:.1f}%")
    print(f"  → Recommended   : {direction}")
    return confidences, direction


# ==============================
# MAIN LOOP
# ==============================


print(f"Output folder: {OUTPUT_PATH}\n")


if os.path.isdir(INPUT_PATH):
    file_list = [f for f in sorted(os.listdir(INPUT_PATH)) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    for filename in file_list:
        image_path = os.path.join(INPUT_PATH, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Skipping {filename} — could not read.")
            continue
        print(f"Processing {filename}...")
        process_image(image, filename)
        print()
elif os.path.isfile(INPUT_PATH):
    filename = os.path.basename(INPUT_PATH)
    image = cv2.imread(INPUT_PATH)
    if image is None:
        print(f"  Skipping {filename} — could not read.")
    else:
        print(f"Processing {filename}...")
        process_image(image, filename)
        print()
else:
    print(f"INPUT_PATH '{INPUT_PATH}' is not a valid file or directory.")

print("Done.")