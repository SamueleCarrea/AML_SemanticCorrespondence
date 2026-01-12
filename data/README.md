# SPair-71k Dataset Setup

## Overview

SPair-71k is a large-scale benchmark dataset for semantic correspondence, containing 70,958 image pairs across 18 object categories with manually annotated keypoint correspondences.

---

## Dataset Structure

Place the Spair-71k dataset under this `dataset/` directory with the following structure:

```
dataset/
└── Spair-71k/
    ├── JPEGImages/              # All source and target images
    │   ├── aeroplane/
    │   ├── bicycle/
    │   ├── bird/
    │   └── ... (18 categories total)
    │
    ├── ImageAnnotation/         # Keypoint annotations per split
    │   ├── train/
    │   │   └── pairs.json       # Training pairs with keypoints
    │   ├── val/
    │   │   └── pairs.json       # Validation pairs
    │   └── test/
    │       └── pairs.json       # Test pairs (1,814 pairs)
    │
    ├── PairAnnotation/          # Original category-wise annotations (optional)
    │   ├── aeroplane/
    │   ├── bicycle/
    │   └── ...
    │
    ├── ImageSets/               # Split files (optional, for reference)
    │   └── main/
    │       ├── train.txt
    │       ├── val.txt
    │       └── test.txt
    │
    ├── keypoints/               # Keypoint definitions per category
    │   ├── aeroplane.json
    │   ├── bicycle.json
    │   └── ... (18 files)
    │
    └── symmetry.txt             # Symmetric keypoint mappings (optional)
```

---

## Dataset Statistics

| Category    | Train Pairs | Val Pairs | Test Pairs | Total Images |
|-------------|-------------|-----------|------------|--------------|
| aeroplane   | 400         | 100       | 100        | 600          |
| bicycle     | 400         | 100       | 100        | 600          |
| bird        | 400         | 100       | 100        | 600          |
| boat        | 400         | 100       | 100        | 600          |
| bottle      | 400         | 100       | 100        | 600          |
| bus         | 400         | 100       | 100        | 600          |
| car         | 400         | 100       | 100        | 600          |
| cat         | 400         | 100       | 100        | 600          |
| chair       | 400         | 100       | 100        | 600          |
| cow         | 400         | 100       | 100        | 600          |
| dog         | 400         | 100       | 100        | 600          |
| horse       | 400         | 100       | 100        | 600          |
| motorbike   | 400         | 100       | 100        | 600          |
| person      | 400         | 100       | 100        | 600          |
| pottedplant | 400         | 100       | 100        | 600          |
| sheep       | 400         | 100       | 100        | 600          |
| train       | 400         | 100       | 100        | 600          |
| tvmonitor   | 400         | 100       | 100        | 600          |
| **TOTAL**   | **7,200**   | **1,800** | **1,814**  | **10,814**   |

---

## Annotation Format

### `ImageAnnotation/<split>/pairs.json`

Each JSON file contains a list of image pairs with keypoint annotations:

```json
[
  {
    "pair_id": "aeroplane_0001",
    "category": "aeroplane",
    "src_img": "aeroplane/2008_000033.jpg",
    "tgt_img": "aeroplane/2008_000042.jpg",
    "src_kps": [[x1, y1], [x2, y2], ...],  // (N, 2) pixel coordinates
    "tgt_kps": [[x1, y1], [x2, y2], ...],  // (N, 2) pixel coordinates
    "valid": [true, false, true, ...],     // (N,) visibility mask
    "src_bbox": [xmin, ymin, xmax, ymax],  // bounding box (optional)
    "tgt_bbox": [xmin, ymin, xmax, ymax]
  },
  ...
]
```

### `keypoints/<category>.json`

Defines semantic keypoint names for each category:

```json
{
  "keypoints": [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "neck", "left_shoulder", "right_shoulder", ...
  ]
}
```

---

## Usage

### Basic Loading

```python
from dataset.spair import SPairDataset
from torch.utils.data import DataLoader

# Load test split
test_dataset = SPairDataset(
    root="dataset/Spair-71k",
    split="test",
    long_side=518,      # Resize long side
    normalize=True       # Apply ImageNet normalization
)

# Create DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2
)

# Iterate
for batch in test_loader:
    src_img = batch['src_img']      # (B, 3, H, W)
    tgt_img = batch['tgt_img']      # (B, 3, H, W)
    src_kps = batch['src_kps']      # (B, N, 2)
    tgt_kps = batch['tgt_kps']      # (B, N, 2)
    valid_mask = batch['valid_mask'] # (B, N)
    category = batch['category']     # List[str]
    
    # Your processing...
```

### Advanced Options

```python
# Training with data augmentation
train_dataset = SPairDataset(
    root="dataset/Spair-71k",
    split="train",
    long_side=480,
    normalize=True,
    augment=True  # Random flip, crop, etc.
)

# Validation without augmentation
val_dataset = SPairDataset(
    root="dataset/Spair-71k",
    split="val",
    long_side=480,
    normalize=True,
    augment=False
)
```

---

## Evaluation Metrics

SPair-71k uses **Percentage of Correct Keypoints (PCK)** as the primary metric:

### PCK@α Definition

A predicted keypoint is considered correct if:

```
||pred_kp - gt_kp||₂ ≤ α × max(H, W)
```

where:
- `α` is the normalized distance threshold (typically 0.05, 0.10, 0.15, 0.20)
- `max(H, W)` is the maximum image dimension

### Standard Thresholds

| Metric    | Threshold | Description                    |
|-----------|-----------|--------------------------------|
| PCK@0.05  | 5%        | Very strict (high precision)   |
| PCK@0.10  | 10%       | **Primary metric** (standard)  |
| PCK@0.15  | 15%       | Moderate tolerance             |
| PCK@0.20  | 20%       | Relaxed (high recall)          |

### Evaluation Example

```python
from dataset.spair import compute_pck

# Predict keypoints
pred_kps = model.predict(src_img, tgt_img, src_kps)

# Compute PCK
pck_results = compute_pck(
    pred_kps=pred_kps,
    gt_kps=tgt_kps,
    image_size=(H, W),
    thresholds=[0.05, 0.10, 0.15, 0.20]
)

print(f"PCK@0.10: {pck_results['PCK@0.10']:.4f}")
```

---

## Notes

### Symmetric Keypoints

Some categories have symmetric keypoints (e.g., "left_eye" ↔ "right_eye"). The `symmetry.txt` file defines these mappings, useful for:
- Data augmentation (horizontal flip)
- Handling symmetric objects
- Improving robustness

### Category-Specific Keypoints

Different categories have different numbers of keypoints:
- **person**: 17 keypoints (COCO format)
- **aeroplane**: 12 keypoints
- **bicycle**: 10 keypoints
- **cat/dog**: 15 keypoints
- etc.

Always check `valid_mask` to filter out invisible/occluded keypoints.

### Image Sizes

- Original images have varying resolutions
- Common practice: resize so that `max(H, W) = 480` or `518`
- Maintain aspect ratio during resize
- Keypoint coordinates scale proportionally

---

## Citation

If you use SPair-71k in your research, please cite:

```bibtex
@inproceedings{min2019spair,
  title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
  author={Min, Juhong and Lee, Jongmin and Ponce, Jean and Cho, Minsu},
  booktitle={arXiv:1908.10543},
  year={2019}
}
```

---

## Troubleshooting

### Dataset not found
```python
# Check path
from pathlib import Path
assert Path("dataset/Spair-71k").exists(), "Dataset not found!"

# List contents
!ls -la dataset/Spair-71k/
```

### Missing annotations
```python
# Verify annotation files exist
import json
with open("dataset/Spair-71k/ImageAnnotation/test/pairs.json") as f:
    pairs = json.load(f)
print(f"Found {len(pairs)} test pairs")
```

### Memory issues
```python
# Use smaller batch size or fewer workers
loader = DataLoader(
    dataset,
    batch_size=1,      # Reduce if OOM
    num_workers=0,     # Set to 0 if multiprocessing errors
    pin_memory=False   # Disable if low on memory
)
```

---

## License

SPair-71k is released under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

For commercial use, contact the original authors.

---

**Last Updated:** January 2026  
**Maintained by:** AML Semantic Correspondence Project
