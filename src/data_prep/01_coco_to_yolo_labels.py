"""
Milestone 1: COCO -> YOLO label conversion (localization-only, single class)

Converts train_quadrant_enumeration_disease.json (COCO-style, per-box
quadrant/tooth-number/diagnosis labels) into YOLO-format .txt label files,
collapsing all 4 diagnosis categories into ONE class: "abnormal_candidate".

Why single class: YOLO's job in our two-stage pipeline is localization only.
Diagnosis (which of the 4 classes) is Swin-Tiny's job downstream, on the
cropped regions YOLO finds. So category_id_3 (diagnosis) is intentionally
NOT used as the YOLO class here -- every box just becomes class 0.

Run this AFTER split_dataset.py (Milestone 0) -- it reads train.txt/val.txt/
test.txt to know which images go in which split.
"""

import json
from pathlib import Path
from collections import defaultdict

import json
from pathlib import Path
from collections import defaultdict

# Make the repo root importable regardless of where this script is run from,
# since it lives two folders deep (src/data_prep/) but config.py sits at the
# repo root. __file__ is this script's own path; .resolve() makes it
# absolute; .parents[2] walks up: data_prep -> src -> repo root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[2]))

from config import JSON_PATH, SPLITS_DIR, LABELS_DIR

# ---- config ----
OUT_DIR = LABELS_DIR

# ---- load annotations ----
with open(JSON_PATH) as f:
    data = json.load(f)

# map image_id -> (file_name, width, height)  -- we need width/height to normalize coordinates
images_by_id = {img["id"]: img for img in data["images"]}

# group all boxes by image_id (an image can have multiple abnormal teeth)
boxes_by_image = defaultdict(list)
for ann in data["annotations"]:
    boxes_by_image[ann["image_id"]].append(ann["bbox"])  # COCO bbox: [x_min, y_min, w, h]

# ---- load which file_name belongs to which split ----
def load_split_filenames(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

splits = {
    "train": load_split_filenames(SPLITS_DIR / "train.txt"),
    "val": load_split_filenames(SPLITS_DIR / "val.txt"),
    "test": load_split_filenames(SPLITS_DIR / "test.txt"),
}

# build a quick lookup: file_name -> which split it's in
filename_to_split = {}
for split_name, filenames in splits.items():
    for fn in filenames:
        filename_to_split[fn] = split_name

# ---- convert and write ----
for split_name in splits:
    (OUT_DIR / split_name).mkdir(parents=True, exist_ok=True)

written = 0
skipped_no_split = 0

for img_id, img_info in images_by_id.items():
    file_name = img_info["file_name"]
    split_name = filename_to_split.get(file_name)

    if split_name is None:
        # image wasn't assigned to any split (shouldn't normally happen if
        # split_dataset.py covered all images) -- skip and count it
        skipped_no_split += 1
        continue

    img_w, img_h = img_info["width"], img_info["height"]
    lines = []

    for bbox in boxes_by_image.get(img_id, []):  # empty list if image has 0 annotations
        x_min, y_min, box_w, box_h = bbox

        # convert COCO [x_min, y_min, w, h] in PIXELS
        # to YOLO [x_center, y_center, w, h] as FRACTIONS of image size (0-1)
        x_center = (x_min + box_w / 2) / img_w
        y_center = (y_min + box_h / 2) / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h

        class_id = 0  # single class: "abnormal_candidate" -- see module docstring
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    # write one .txt file per image, matching the image's base filename.
    # NOTE: images with zero annotations get an EMPTY .txt file -- this is
    # correct and required. YOLO treats a missing detection target as a
    # negative/background example, which we WANT (see split_dataset.py notes
    # on the 27 zero-annotation images).
    label_filename = Path(file_name).stem + ".txt"
    with open(OUT_DIR / split_name / label_filename, "w") as f:
        f.write("\n".join(lines))

    written += 1

print(f"Wrote {written} label files across train/val/test.")
if skipped_no_split:
    print(f"WARNING: {skipped_no_split} images had no split assignment and were skipped. "
          f"Check that split_dataset.py ran on the same JSON file.")

# ---- quick sanity check: print one example ----
example_split = "train"
example_files = sorted((OUT_DIR / example_split).glob("*.txt"))
if example_files:
    print(f"\nExample label file: {example_files[0].name}")
    print(example_files[0].read_text() or "(empty -- this image has no abnormal teeth)")
