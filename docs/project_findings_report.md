# Trust-Aware Explainable AI for Abnormal Tooth Detection and Diagnosis in Panoramic Dental X-Rays

## Project Findings & Design Report
*Compiled for reference during paper writing. Captures dataset analysis, all architectural decisions, and the reasoning behind each.*

---

## 1. Dataset

**Source:** DENTEX Challenge 2023 (Kaggle mirror: `truthisneverlinear/dentex-challenge-2023`)

The DENTEX dataset is released as three subsets with progressively richer annotations, and — importantly — these subsets contain **different, non-overlapping images**, not the same images with added labels:

| Subset | Images | Annotations | Labels present |
|---|---|---|---|
| Quadrant only | 693 | — | Quadrant (1–4) |
| Quadrant + Enumeration | 634 | 18,095 | Quadrant + tooth number (FDI system) — **every visible tooth boxed** |
| **Quadrant + Enumeration + Diagnosis** | **705** | **3,529** | Quadrant + tooth number + **diagnosis** — **only abnormal teeth boxed** |

The project uses the third subset (`train_quadrant_enumeration_disease.json`) as its primary dataset, since it's the only one with diagnosis labels — the actual detection/diagnosis target.

**Annotation format:** COCO-style JSON with three parallel category fields per annotation object:
- `category_id_1` → quadrant
- `category_id_2` → tooth number (FDI)
- `category_id_3` → **diagnosis** (the label used for this project)

**Diagnosis classes and distribution (annotation-level, n=3529):**

| Diagnosis | Count | Share |
|---|---|---|
| Caries | 2189 | 62.0% |
| Impacted | 604 | 17.1% |
| Deep Caries | 578 | 16.4% |
| **Periapical Lesion** | **158** | **4.5%** |

**Key findings that shaped design decisions:**
- Significant class imbalance — Periapical Lesion is ~14x rarer than Caries. This drove the decision to use **class-weighted loss** during classifier training and to require **stratified splitting** rather than random splitting.
- **605 of 705 images (86%) contain more than one annotation**, and **458 of 705 (65%) contain more than one diagnosis type in the same image**. This makes the dataset a **multi-label problem at the image level**, which is why a standard single-label stratified split (`sklearn.train_test_split(stratify=...)`) is inappropriate — it only supports one label per sample.
- **27 of 705 images have zero annotations** (no abnormal teeth found) — retained in the dataset as negative/background examples, distributed proportionally across splits.

**Why DENTEX over larger scraped alternatives (e.g. Roboflow Universe datasets with 10k–14k images):**
- DENTEX originates from a MICCAI 2023 challenge with published baselines — enables direct benchmark comparison in the paper.
- Larger Roboflow Universe alternatives surveyed had no published description, 0 stars/downloads, and unclear annotation provenance/quality — a risk for a diagnosis task where label precision directly affects trust-related metrics.
- Data augmentation (see §3) addresses the small-dataset concern without needing to switch datasets, and is standard practice regardless of dataset size.

---

## 2. Dataset Splitting

**Method:** Multi-label stratified split via **iterative stratification** (`iterstrat` / `MultilabelStratifiedShuffleSplit`), not a standard single-label stratified split — necessary because of the multi-label-per-image structure described above.

**Split ratios:** 80% train / 10% val / 10% test

**Resulting split (achieved balance across all 4 diagnosis classes, including the rare Periapical Lesion):**

| Split | Images | Notes |
|---|---|---|
| Train | 560 | ~79.3% — used for training only |
| Val | 70 | ~10.3% — model selection / early stopping |
| Test | 75 | ~10.3% — held out, touched only for final evaluation |

Each split closely preserves the overall class proportions (including Periapical Lesion), verified via a generated `split_report.txt` documenting per-class instance counts in each split — this file itself is citable as evidence of stratification quality in the paper's methodology section.

**Rule established:** the test set is never touched until final evaluation. Augmentation (see §3) and any preprocessing tuning happens only against train/val.

---

## 3. Preprocessing and Augmentation

### 3.1 Preprocessing (deterministic, applied once, saved)

Applied consistently to all images regardless of split:

1. **Resize** — panoramic X-rays are large and non-square (~2700×1300px); resized to a fixed input size appropriate to the model stage (detection input size for YOLO; 224×224 for Swin-Tiny crops).
2. **Channel handling** — X-rays are grayscale in content; if stored single-channel, replicated across R/G/B to match the 3-channel input expected by pretrained detection/classification backbones.
3. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — applied to correct uneven/low contrast common in panoramic X-rays, which otherwise makes small lesions (e.g. early caries, periapical lesions) hard to distinguish. This is a domain-specific step drawn from dental/medical X-ray imaging literature, not a generic CV default.
4. **Annotation format conversion** — COCO JSON → the format required by each pipeline stage (see §5).

### 3.2 Augmentation (stochastic, applied live, train set only)

**Library:** Albumentations — chosen specifically because it is bounding-box-aware, i.e. it automatically recalculates box coordinates when an image is rotated/flipped, avoiding manual (error-prone) coordinate math.

**Mode:** **Online augmentation** — applied fresh, in memory, every time an image is loaded during training; nothing is saved to disk. Chosen over offline augmentation (pre-generating and saving N augmented copies) because:
- Offline augmentation produces a fixed, finite set of variations that a model can begin to memorize over many epochs; online augmentation gives effectively unlimited variation.
- No extra disk/storage overhead — relevant on Kaggle's storage-constrained notebook environment.
- Standard, built-in behavior in the Ultralytics (YOLO) training pipeline.

**Scope rule:** augmentation applied **only to the training split**. Validation and test sets remain untouched/unaltered, since they exist to measure performance on realistic, unaltered images. Augmentation is applied strictly *after* the train/val/test split (never before), to avoid leaking near-duplicate augmented images across splits.

**Transforms used (rationale for each):**

| Transform | Included? | Rationale |
|---|---|---|
| Small-angle rotation (±10–15°) | ✅ | Patients aren't always perfectly aligned in the scanner |
| Horizontal flip | ✅ | Anatomically valid (left/right quadrants mirror) — **requires swapping quadrant labels on flip** (Q1↔Q2, Q3↔Q4 in FDI system) |
| Brightness/contrast jitter | ✅ | Exposure varies across the 3 pooled source hospitals in DENTEX |
| Slight scale/zoom variation | ✅ | Generalizes over framing differences |
| Gaussian noise / mild blur | ✅ | Simulates varying equipment quality |
| Vertical flip | ❌ | Anatomically invalid — X-rays are never upside down in practice |
| Large rotation (>20°) | ❌ | Panoramic X-rays have a fixed anatomical layout; unrealistic beyond small angles |
| Heavy color/hue jitter | ❌ | Images are grayscale-derived; aggressive RGB color augmentation doesn't apply |
| Cutout / random erasing | ❌ | Risks erasing the exact abnormal region the model needs to learn from |

**Estimated training time cost of online augmentation:** overlaps with GPU compute via background CPU worker processes (Ultralytics default behavior), typically adding only ~10–20% overhead versus no augmentation. Estimated full training run: ~1.5–2.5 min/epoch on Kaggle T4/P100 at 560 training images, batch size 16 → roughly 2.5–4 hours for a ~100-epoch run. Well within Kaggle's ~30 GPU-hrs/week free quota.

---

## 4. Architecture — Evolution and Final Design

### 4.1 Initial design (superseded)
Originally planned as a **single-stage** approach: one YOLO model performing both localization (bounding box) and diagnosis classification simultaneously, using `category_id_3` as the YOLO class label.

### 4.2 Final design (current): Two-stage pipeline

```
Panoramic X-ray
      ↓
YOLO Tooth Detector (localization only, single class: "abnormal candidate")
      ↓
Crop each detected region (+ padding margin)
      ↓
Swin-Tiny Classifier (4-way diagnosis)
      ↓
Dual XAI: Grad-CAM++ + Integrated Gradients (via Captum)
      ↓
Faithfulness/localization check: Pointing Game + IoU against ground-truth boxes
      ↓
Coordinate mapping: crop-local heatmaps → original panoramic image space
      ↓
Annotated Diagnostic Report
```

**Reasons for switching from single-stage to two-stage (as reasoned through in project discussion):**

1. **Research contribution alignment.** The project's explainability/faithfulness contribution is specifically about the *diagnostic classifier*, which is more natural to study on a dedicated classifier (Swin-Tiny) than on a detector's classification head.
2. **XAI tooling maturity.** Grad-CAM++ and Integrated Gradients (via Captum) are natively designed for standard classifiers — a single image in, single class-probability vector out. Applying them to YOLO directly is comparatively awkward, since YOLO's raw output is a dense, multi-object, per-anchor prediction grid rather than a clean class-score vector; using them on YOLO would require custom wrapper code. Routing diagnosis through Swin-Tiny gets both XAI methods essentially "for free" from mature, well-tested library implementations.
3. **Modularity.** Detection and classification stages can be independently improved, retrained, or swapped later without redesigning the whole pipeline.
4. **Cleaner interpretability scope.** Each Grad-CAM++/Integrated Gradients heatmap is scoped to a single cropped tooth, rather than a whole panoramic image containing multiple detected objects — conceptually and visually cleaner for the explainability analysis.

**Technical trade-offs accepted knowingly (documented, not overlooked):**

- **Error propagation** — a missed or poorly-boxed detection from YOLO caps what Swin-Tiny can ever classify correctly. The paper's evaluation should report end-to-end performance (detection recall × conditional classification accuracy), not just Swin-Tiny's standalone accuracy on ground-truth crops.
- **Crop quality dependency** — box tightness matters; too tight crops off diagnostic context (e.g. root apex for periapical lesion judgment), too loose introduces noise from neighboring teeth/gum tissue. A padding margin (10–20%) around each YOLO box before cropping was adopted as a tunable setting, not a fixed constant.
- **Swin-Tiny (Vision Transformer) is more data-hungry than a CNN** — ViT-family models lack CNNs' built-in spatial inductive biases and typically need more data or stronger reliance on pretraining. With only ~125 training crops for the rarest class (Periapical Lesion) after splitting, mitigations planned: ImageNet-pretrained Swin-Tiny (transfer learning, not training from scratch), strong augmentation on crops, class-weighted loss.
- **Two full training pipelines instead of one** — more engineering/debugging surface area within the 2–3 week timeline; mitigated by running a tiny-subset debug pass on *both* stages before committing real Kaggle GPU hours to either.
- **Coordinate-mapping step required** — to reconstruct the final annotated report (heatmaps overlaid on the original full panoramic image), each crop's local heatmap coordinates must be mapped back to their position in the original image.

**Dataset scope decision for YOLO (explicitly settled):** because the diagnosis-labeled subset (705 images) only contains bounding boxes for *abnormal* teeth (not healthy ones), and the fully-boxed quadrant-enumeration subset (634 images, every tooth boxed) does not share overlapping images with the diagnosis subset, YOLO is scoped as an **abnormal-candidate-region detector** (single class), trained on the diagnosis subset's boxes. It does **not** attempt full-mouth (healthy + abnormal) tooth detection, since that would require fabricated/unavailable "healthy crop" ground truth. This directly matches the project's stated scope: "abnormal tooth detection and diagnosis," not general dental charting.

**Optional pretraining strategy:** YOLO's localization ability may be pretrained on the larger 634-image quadrant-enumeration subset (general tooth-shape prior, since every tooth is boxed there) before fine-tuning on the actual 705-image abnormal-region boxes it needs to detect. This is a legitimate way to extract more value from the DENTEX release beyond the primary 705-image subset.

**Training/evaluation data split for Swin-Tiny (standard two-stage detect-then-classify practice, adopted):**
- **Train** Swin-Tiny on **ground-truth crops** (correctly positioned, from annotation boxes) — so it learns real diagnostic features rather than YOLO's localization errors.
- **Evaluate the full end-to-end pipeline** using YOLO's **predicted** crops — so final reported numbers reflect real-world performance including YOLO's imperfections, not an inflated ground-truth-only number.

---

## 5. Model Selection

### 5.1 Detection stage: YOLOv8 vs YOLOv11 (comparative analysis)

**Rationale for the YOLO family generally (over e.g. Faster R-CNN/Detectron2):**

| Criterion | YOLO (Ultralytics) | Faster R-CNN (Detectron2) |
|---|---|---|
| Beginner-friendliness | High — concise API, strong docs | Lower — more config surface |
| Training speed on Kaggle GPU | Fast, fits GPU-hour budget | Slower, consumes more of the 30 hr/week quota |
| Performance on small datasets | Strong — built-in augmentation, anchor-free design | Workable but typically needs more data/tuning |
| Used in dental detection literature | Common, including DENTEX-adjacent work | Also used, including original DENTEX baselines |

**Comparison protocol:** YOLOv8 and YOLOv11 are trained under **identical conditions** — same train/val split, same preprocessing, same augmentation config — so the comparison isolates architecture differences. This is a **one-time offline experiment**, run once during development (not something re-run per inference).

**Comparison metrics:** see §6.1.

**After comparison:** only the winning model is carried forward into every subsequent stage (crop generation, Swin-Tiny training, XAI, demo). Both models are never run simultaneously at inference/deployment time — the losing model is fully retired after the Week 2 comparison.

### 5.2 Classification stage: Swin-Tiny

Chosen for diagnosis classification on cropped tooth images. ImageNet-pretrained, fine-tuned via transfer learning (not trained from scratch, given the small per-class crop counts). Vision Transformer architecture — see §4.2 for associated data-hungriness trade-off and mitigations.

---

## 6. Evaluation Metrics (finalized)

### 6.1 YOLOv8 vs YOLOv11 (detection/localization comparison)

| Metric | Purpose |
|---|---|
| mAP@0.5 | Headline detection benchmark number |
| mAP@0.5:0.95 | Stricter, COCO-style localization quality |
| Precision, Recall, F1 | Standard detection quality measures |
| Inference speed (ms/image or FPS) | Practical deployability comparison |
| Model size / parameter count | Deployment footprint comparison |

*(Note: since YOLO is now single-class "abnormal candidate" rather than multi-class diagnosis, per-class breakdown at this stage is not applicable — per-class performance is instead measured downstream at the Swin-Tiny classification stage.)*

### 6.2 Swin-Tiny diagnosis classifier

| Metric | Purpose |
|---|---|
| Per-class accuracy/precision/recall/F1 | Diagnosis quality per class, especially Periapical Lesion (imbalance-sensitive) |
| Confusion matrix | Which diagnoses get confused with which |

### 6.3 Grad-CAM++ vs Integrated Gradients (XAI comparison)

| Metric | Purpose |
|---|---|
| **Pointing Game** | Does the heatmap's peak activation fall inside the ground-truth box? (% accuracy across test images) |
| **IoU-based localization** | Overlap between thresholded heatmap region and ground-truth box |
| Inter-method agreement | Overlap (IoU/cosine similarity) between Grad-CAM++ and Integrated Gradients maps on the same image — convergence between methods as a trust signal |
| Explanation generation time | Practical speed trade-off (Grad-CAM++: single backward pass, fast; Integrated Gradients: multi-step path integral, slower) |
| Qualitative side-by-side panels | Visual sanity check — does highlighted region land on the actual lesion/tooth? |

**Explicitly excluded (scope decision):** insertion/deletion/ROAD perturbation-based faithfulness metrics, and average-pixel-drop-style measures — cut for time, in favor of the ground-truth-box-based Pointing Game/IoU metrics above, which are cheaper to compute (no perturbation/re-inference required) since ground-truth boxes are already available in the dataset.

### 6.4 Trust/Calibration layer

| Metric | Purpose |
|---|---|
| Expected Calibration Error (ECE) | Quantifies gap between predicted confidence and actual accuracy |
| Reliability diagrams | Visual calibration check |
| Confidence histograms | Distribution of model confidence across predictions |

Calibration method: temperature scaling (planned) applied to Swin-Tiny's softmax outputs.

---

## 7. Tools and Environment

| Purpose | Tool |
|---|---|
| Compute environment | Kaggle Notebooks (GPU: T4/P100, ~30 hrs/week free quota) |
| Detection framework | Ultralytics (YOLOv8, YOLOv11) |
| Classification framework | PyTorch (Swin-Tiny, torchvision/timm pretrained weights) |
| Augmentation | Albumentations (bounding-box-aware transforms) |
| Multi-label stratified splitting | `iterative-stratification` (`iterstrat` package) |
| Explainability | Captum (Grad-CAM++, Integrated Gradients) |
| Demo app (planned) | Streamlit or Gradio |

**Why Kaggle over Google Colab:** the dataset is already hosted as a Kaggle Dataset and attaches to a Kaggle Notebook with no download/API-key step; Colab would require manual Kaggle API token setup and re-downloading the dataset every session (Colab storage doesn't persist between sessions). No GPU-quota advantage to Colab for this use case.

---

## 8. Project Timeline (2–3 week scope)

**Week 1 — Data + localization baseline**
- COCO parsing, class distribution analysis (done — see §1)
- Stratified multi-label train/val/test split (done — see §2)
- Preprocessing pipeline: resize + CLAHE
- Augmentation pipeline: Albumentations, train-only
- Single-class relabeling of boxes for YOLO localization task
- Tiny-subset debug run (pipeline sanity check, minimal GPU cost)
- YOLOv8 vs YOLOv11 training + comparison → select winning detector

**Week 2 — Classification + explainability**
- Ground-truth crop generation (with padding margin) for Swin-Tiny training
- Swin-Tiny training (4-way diagnosis, class-weighted loss)
- Grad-CAM++ implementation (Captum)
- Integrated Gradients implementation (Captum)
- Pointing Game / IoU evaluation, inter-method agreement, generation-time comparison

**Week 3 — Trust layer + full evaluation + packaging**
- Calibration: temperature scaling, ECE, reliability diagrams
- Full end-to-end evaluation using YOLO's *predicted* crops (not ground truth)
- Coordinate-mapping step: crop-local heatmaps → original image space
- Demo app (Streamlit/Gradio): X-ray → detection → diagnosis → confidence → heatmap
- Write-up: methodology, YOLOv8-vs-v11 comparison results, Grad-CAM++-vs-IG comparison results, limitations

---

## 9. Chronological Decision Log

*Useful for the paper's methodology section — shows the reasoning trail, not just the final choices.*

1. Confirmed dataset scope: used the quadrant-enumeration-**diagnosis** subset specifically (not the quadrant-only or quadrant-enumeration subsets), since it's the only one with diagnosis labels.
2. Identified no official test set is usable (Kaggle mirror's validation images are unlabeled, true 250-image test set is not public) → decided to carve train/val/test from the 705 labeled images directly.
3. Identified multi-label structure (images with multiple diagnosis types) → adopted iterative/multi-label stratification instead of standard single-label stratified split.
4. Confirmed significant class imbalance (Periapical Lesion 4.5% of annotations) → informed decisions on class-weighted loss and stratification necessity.
5. Decided on CLAHE preprocessing specifically for X-ray contrast correction (domain-specific, not generic CV default).
6. Decided on Albumentations for bounding-box-aware augmentation.
7. Decided on **online** augmentation over offline, and restricted augmentation to the training split only, applied after splitting.
8. Considered switching to a larger Roboflow Universe dataset to sidestep augmentation needs → rejected: augmentation isn't a workaround for small data (used regardless of size), and DENTEX's academic benchmark backing outweighs raw image count from an unverified source.
9. Added YOLOv8-vs-YOLOv11 comparative analysis to the plan.
10. Added Grad-CAM++-vs-Integrated-Gradients comparative analysis to the plan.
11. Removed insertion/deletion/ROAD faithfulness metrics from scope (time trade-off).
12. Clarified that the YOLOv8/v11 comparison is a one-time offline experiment; only the winning model is used at deployment/inference time going forward.
13. **Major architecture revision:** switched from single-stage (YOLO doing joint detection+diagnosis) to two-stage (YOLO localization-only → crop → Swin-Tiny diagnosis), driven by the explainability research focus and XAI tooling compatibility.
14. Resolved dataset-driven scope question: YOLO detects abnormal-candidate regions only (single class), not full-mouth teeth, since healthy-tooth ground truth isn't available in a form matched to the diagnosis-labeled images.
15. Re-confirmed "Faithfulness Evaluation" in the new pipeline refers to the already-scoped Pointing Game/IoU metrics, not a reintroduction of insertion/deletion/ROAD.

---

## 10. Open Items (not yet decided — flag for future sessions)

- Exact padding margin percentage for crop generation (to be tuned empirically)
- Exact Swin-Tiny hyperparameters (learning rate, weight decay, dropout) — pending Week 2 experimentation
- Whether YOLO pretraining on the quadrant-enumeration subset will be used, or skipped for time
- Final demo app framework choice (Streamlit vs Gradio) — functionally interchangeable, not yet decided
- COCO → YOLO / crop-generation conversion scripts — planned next coding step
