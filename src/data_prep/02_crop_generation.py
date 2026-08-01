"""
Milestone 2: Crop generation for Swin-Tiny diagnosis classifier

For each annotated abnormal tooth, crops the region out of its panoramic
X-ray (with a padding margin for diagnostic context), and saves it as an
individual image file organized by split and diagnosis class:

    crops/train/Caries/train_0_ann1.png
    crops/train/Impacted/train_5_ann12.png
    crops/val/...
    crops/test/...

Also writes crop_metadata.csv recording, for every crop: which original
image it came from, its original (un-padded) pixel bbox, and its padded
pixel bbox. This is required later (Week 3) to map Grad-CAM++/Integrated
Gradients heatmaps -- which are computed on the crop -- back onto their
correct position in the original panoramic X-ray for the final report.

Run this AFTER split_dataset.py. Independent of coco_to_yolo_labels.py
(both are downstream of the split, not of each other).
"""

import csv
import json
from pathlib import Path

from PIL import Image

import csv
import json
from pathlib import Path

from PIL import Image

# Make the repo root importable regardless of where this script is run from,
# since it lives two folders deep (src/data_prep/) but config.py sits at the
# repo root. __file__ is this script's own path; .resolve() makes it
# absolute; .parents[2] walks up: data_prep -> src -> repo root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[2]))

from config import JSON_PATH, IMAGES_DIR, SPLITS_DIR, CROPS_DIR

# ---- config ----
OUT_DIR = CROPS_DIR
PAD_FRAC = 0.15   # expand each box by 15% on every side before cropping

# ---- load annotations ----
with open(JSON_PATH) as f:
    data = json.load(f)

images_by_id = {img["id"]: img for img in data["images"]}
diag_map = {c["id"]: c["name"] for c in data["categories_3"]}

def load_split_filenames(path):
    with open(path) as f:
        return set(line.strip() for line in f if line.strip())

splits = {
    "train": load_split_filenames(SPLITS_DIR / "train.txt"),
    "val": load_split_filenames(SPLITS_DIR / "val.txt"),
    "test": load_split_filenames(SPLITS_DIR / "test.txt"),
}
filename_to_split = {fn: s for s, fnames in splits.items() for fn in fnames}

# folder-safe class names (spaces aren't safe in directory names on all systems)
def safe_class_name(name):
    return name.replace(" ", "_")

for split_name in splits:
    for cid, name in diag_map.items():
        (OUT_DIR / split_name / safe_class_name(name)).mkdir(parents=True, exist_ok=True)

# ---- crop generation ----
# Group annotations by image_id FIRST, so each source image is opened and
# decoded from disk exactly ONCE, no matter how many abnormal teeth it
# contains. The earlier version of this script opened the image once PER
# ANNOTATION -- since 605/705 images have multiple annotations, that meant
# redundantly re-decoding the same large panoramic X-ray file up to several
# times each. With 3529 annotations across 705 images, that's up to 5x more
# disk reads/decodes than necessary, which is almost certainly why the
# unoptimized version was taking 10+ minutes.
from collections import defaultdict

anns_by_image = defaultdict(list)
for ann in data["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

metadata_rows = []
written, skipped_missing_image, skipped_no_split = 0, 0, 0

for image_id, anns in anns_by_image.items():
    img_info = images_by_id[image_id]
    file_name = img_info["file_name"]
    split_name = filename_to_split.get(file_name)

    if split_name is None:
        skipped_no_split += len(anns)
        continue

    img_path = Path(IMAGES_DIR) / file_name
    if not img_path.exists():
        skipped_missing_image += len(anns)
        continue

    img_w, img_h = img_info["width"], img_info["height"]

    # open + decode this source image exactly once, then crop every one of
    # its annotations from that single decoded copy
    with Image.open(img_path) as im:
        for ann in anns:
            x_min, y_min, box_w, box_h = ann["bbox"]  # COCO: pixel x_min, y_min, width, height

            pad_w = box_w * PAD_FRAC
            pad_h = box_h * PAD_FRAC
            padded_x_min = x_min - pad_w
            padded_y_min = y_min - pad_h
            padded_x_max = x_min + box_w + pad_w
            padded_y_max = y_min + box_h + pad_h

            # clip to image bounds -- padding can push the box outside the
            # image edges (e.g. teeth near the left/right edge of the scan)
            clipped_x_min = max(0, padded_x_min)
            clipped_y_min = max(0, padded_y_min)
            clipped_x_max = min(img_w, padded_x_max)
            clipped_y_max = min(img_h, padded_y_max)

            crop = im.crop((clipped_x_min, clipped_y_min, clipped_x_max, clipped_y_max))

            diagnosis_id = ann["category_id_3"]
            diagnosis_name = safe_class_name(diag_map[diagnosis_id])
            crop_filename = f"{Path(file_name).stem}_ann{ann['id']}.png"
            out_path = OUT_DIR / split_name / diagnosis_name / crop_filename
            crop.save(out_path)

            metadata_rows.append({
                "crop_filename": crop_filename,
                "split": split_name,
                "diagnosis": diag_map[diagnosis_id],
                "original_image": file_name,
                "original_bbox_x": x_min, "original_bbox_y": y_min,
                "original_bbox_w": box_w, "original_bbox_h": box_h,
                "padded_bbox_x_min": clipped_x_min, "padded_bbox_y_min": clipped_y_min,
                "padded_bbox_x_max": clipped_x_max, "padded_bbox_y_max": clipped_y_max,
            })
            written += 1

# ---- write metadata CSV ----
with open(OUT_DIR / "crop_metadata.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
    writer.writeheader()
    writer.writerows(metadata_rows)

print(f"Wrote {written} crops.")
if skipped_missing_image:
    print(f"WARNING: {skipped_missing_image} annotations skipped -- image file not found. "
          f"Check IMAGES_DIR is correct.")
if skipped_no_split:
    print(f"WARNING: {skipped_no_split} annotations skipped -- image not found in any split file.")

print(f"\nCrops per split/class:")
from collections import Counter
counts = Counter((r["split"], r["diagnosis"]) for r in metadata_rows)
for (split_name, diagnosis), count in sorted(counts.items()):
    print(f"  {split_name:6s} {diagnosis:20s}: {count}")
