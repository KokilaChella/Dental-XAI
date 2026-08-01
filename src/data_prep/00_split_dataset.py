"""
DENTEX Challenge 2023 — Quadrant-Enumeration-Diagnosis subset
Class distribution analysis + stratified train/val/test split.

Input:  train_quadrant_enumeration_disease.json  (COCO-like format, 705 images)
Output: splits/train.txt, splits/val.txt, splits/test.txt  (one file_name per line)
        splits/split_report.txt  (class distribution per split, for your report)

Why iterative stratification instead of sklearn's train_test_split(stratify=...):
Each image can contain MULTIPLE abnormal teeth with DIFFERENT diagnoses
(458 of 705 images have mixed diagnosis types). This makes it a multi-label
problem at the image level, not single-label. A plain stratified split
only handles one label per sample and would produce splits with skewed
class balance, especially for rare classes like Periapical Lesion (4.5%
of annotations). Iterative stratification balances every label across
splits simultaneously.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Make the repo root importable regardless of where this script is run from,
# since it lives two folders deep (src/data_prep/) but config.py sits at the
# repo root. __file__ is this script's own path; .resolve() makes it
# absolute; .parents[2] walks up: data_prep -> src -> repo root.
import sys as _sys
from pathlib import Path as _Path
_sys.path.append(str(_Path(__file__).resolve().parents[2]))

from config import JSON_PATH, SPLITS_DIR

# ---- config ----
OUT_DIR = SPLITS_DIR
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.80, 0.10, 0.10
SEED = 42

assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# ---- load ----
with open(JSON_PATH) as f:
    data = json.load(f)

images = data["images"]                      # list of {id, file_name, height, width}
annotations = data["annotations"]             # list of {image_id, category_id_3, ...}
diag_map = {c["id"]: c["name"] for c in data["categories_3"]}
n_classes = len(diag_map)

# ---- build per-image multi-hot label vector ----
img_id_to_labels = defaultdict(set)
for ann in annotations:
    img_id_to_labels[ann["image_id"]].add(ann["category_id_3"])

image_ids = [img["id"] for img in images]
file_names = {img["id"]: img["file_name"] for img in images}

Y = np.zeros((len(image_ids), n_classes), dtype=int)
for row, img_id in enumerate(image_ids):
    for cid in img_id_to_labels.get(img_id, []):
        Y[row, cid] = 1

# images with zero annotations get an all-zero row; IterativeStratification
# handles this fine, they're just distributed proportionally like any other pattern.
X = np.array(image_ids).reshape(-1, 1)

# ---- stratified split: first carve out test, then split remainder into train/val ----
np.random.seed(SEED)

stratifier1 = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=TEST_FRAC, random_state=SEED
)
trainval_idx, test_idx = next(stratifier1.split(X, Y))

X_trainval, Y_trainval = X[trainval_idx], Y[trainval_idx]
val_frac_of_remainder = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)

stratifier2 = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=val_frac_of_remainder, random_state=SEED
)
train_idx_rel, val_idx_rel = next(stratifier2.split(X_trainval, Y_trainval))

train_ids = X_trainval[train_idx_rel].flatten().tolist()
val_ids = X_trainval[val_idx_rel].flatten().tolist()
test_ids = X[test_idx].flatten().tolist()

splits = {"train": train_ids, "val": val_ids, "test": test_ids}

# ---- write file lists ----
OUT_DIR.mkdir(exist_ok=True)
for split_name, ids in splits.items():
    with open(OUT_DIR / f"{split_name}.txt", "w") as f:
        for img_id in ids:
            f.write(file_names[img_id] + "\n")

# ---- report ----
# NOTE: Y only records presence (1/0) per class per image -- it does NOT
# count how many boxes of that class appear in the image. So Y.sum(axis=0)
# is an IMAGE-LEVEL presence count, not a true instance count, and will be
# smaller than the real annotation count whenever an image has more than
# one box of the same class. The stratifier correctly uses presence for
# balancing (that's the standard approach for multi-label stratification),
# but for reporting -- and for planning Milestone 2 crop generation, which
# is per-box, not per-image -- we want the TRUE instance counts too.
true_instance_counts = Counter(a["category_id_3"] for a in annotations)

report_lines = []
report_lines.append(f"Total images: {len(image_ids)}  |  Total annotations: {len(annotations)}")
report_lines.append(f"Images with zero abnormal teeth: {sum(1 for v in Y if v.sum() == 0)}")
report_lines.append("")

overall_presence = Y.sum(axis=0)
report_lines.append("Overall counts (both measures shown -- see note above):")
for cid, name in diag_map.items():
    report_lines.append(
        f"  {name:20s}: {true_instance_counts[cid]:5d} true instances  "
        f"| present in {overall_presence[cid]:4d} images"
    )
report_lines.append(f"  SUM of true instances: {sum(true_instance_counts.values())} (should equal {len(annotations)})")
report_lines.append("")

# per-split TRUE instance counts (this is what Milestone 2 crop generation actually cares about)
per_split_true = defaultdict(Counter)
for ann in annotations:
    for split_name, ids in splits.items():
        if ann["image_id"] in ids:
            per_split_true[split_name][ann["category_id_3"]] += 1
            break

for split_name, ids in splits.items():
    n_imgs = len(ids)
    report_lines.append(f"--- {split_name.upper()}  ({n_imgs} images, {100*n_imgs/len(image_ids):.1f}% of total) ---")
    for cid, name in diag_map.items():
        c = per_split_true[split_name][cid]
        pct_of_class = 100 * c / true_instance_counts[cid] if true_instance_counts[cid] else 0
        report_lines.append(f"  {name:20s}: {c:5d} true instances  ({pct_of_class:.1f}% of all {name} instances)")
    report_lines.append("")

report = "\n".join(report_lines)
print(report)
with open(OUT_DIR / "split_report.txt", "w") as f:
    f.write(report)

print(f"\nSplit files written to {OUT_DIR.resolve()}/ (train.txt, val.txt, test.txt, split_report.txt)")