# SPair-71k Dataset Setup

## Overview

SPair-71k is a large-scale benchmark dataset for semantic correspondence, containing 70,958 image pairs across 18 object categories with manually annotated keypoint correspondences.

---

## Dataset Structure

Place the SPair-71k dataset under this `data/` directory with the following structure:

```
data/
└── SPair-71k/
    ├── ImageAnnotation/         # Json annotation files for all 1,800 images
    │   ├── aeroplane/
    │   ├── bicycle/
    │   ├── bird/
    │   └── ... (18 categories total)
    │
    ├── PairAnnotation/          # Json annotation files for all 70,958 image pairs
    │   ├── aeroplane/           # (53,340 train, 5,384 val, 12,234 test)
    │   ├── bicycle/
    │   ├── bird/
    │   └── ... (18 categories total)
    │
    ├── JPEGImages/              # 1,800 images organized by category
    │   ├── aeroplane/           # 100 images per category
    │   ├── bicycle/
    │   ├── bird/
    │   └── ... (18 categories total)
    │
    ├── Segmentation/            # Segmentation annotations in .png format
    │   ├── aeroplane/
    │   ├── bicycle/
    │   └── ... (18 categories total)
    │
    ├── Layout/                  # Train/val/test splits
    │   ├── large/               # Splits used for evaluation
    │   │   ├── train_pairs.csv
    │   │   ├── val_pairs.csv
    │   │   └── test_pairs.csv
    │   └── small/               # Small subset for faster training/evaluation
    │       ├── train_pairs.csv  # (10,652 training pairs)
    │       ├── val_pairs.csv    # (1,070 validation pairs)
    │       └── test_pairs.csv   # (2,438 test pairs)
    │
    └── devkit/                  # SPair-71k development toolkit
        └── README               # See devkit/README for more info
```

---

## Dataset Statistics

### Images
- **Total Images**: 1,800 (100 per category × 18 categories)
- **Categories**: 18 object classes from PASCAL VOC

### Image Pairs (Large Split)
| Split       | Pairs      | Description                    |
|-------------|------------|--------------------------------|
| Training    | 53,340     | For model training             |
| Validation  | 5,384      | For hyperparameter tuning      |
| Test        | 12,234     | For final evaluation           |
| **TOTAL**   | **70,958** | All annotated pairs            |

### Image Pairs (Small Split - Fast Training)
| Split       | Pairs      | Description                    |
|-------------|------------|--------------------------------|
| Training    | 10,652     | Small subset for faster training |
| Validation  | 1,070      | Small validation subset         |
| Test        | 2,438      | Small test subset               |
| **TOTAL**   | **14,160** | Fast training subset            |

### Categories
| Category    | Images | Pairs (Large) | Typical Keypoints |
|-------------|--------|---------------|-------------------|
| aeroplane   | 100    | ~3,940        | 12                |
| bicycle     | 100    | ~3,940        | 10                |
| bird        | 100    | ~3,940        | 15                |
| boat        | 100    | ~3,940        | 6                 |
| bottle      | 100    | ~3,940        | 9                 |
| bus         | 100    | ~3,940        | 14                |
| car         | 100    | ~3,940        | 20                |
| cat         | 100    | ~3,940        | 15                |
| chair       | 100    | ~3,940        | 10                |
| cow         | 100    | ~3,940        | 15                |
| dog         | 100    | ~3,940        | 15                |
| horse       | 100    | ~3,940        | 15                |
| motorbike   | 100    | ~3,940        | 10                |
| person      | 100    | ~3,940        | 17                |
| pottedplant | 100    | ~3,940        | 8                 |
| sheep       | 100    | ~3,940        | 15                |
| train       | 100    | ~3,940        | 10                |
| tvmonitor   | 100    | ~3,940        | 8                 |

---

## Annotation Format

### `PairAnnotation/<category>/*.json`

Each pair annotation file contains keypoint correspondences:

```json
{
  "src_imname": "2008_000033.jpg",
  "trg_imname": "2008_000042.jpg",
  "category": "aeroplane",
  "src_bndbox": {"xmin": 10, "ymin": 20, "xmax": 300, "ymax": 400},
  "trg_bndbox": {"xmin": 15, "ymin": 25, "xmax": 310, "ymax": 420},
  "src_kps": [[x1, y1], [x2, y2], ...],
  "trg_kps": [[x1, y1], [x2, y2], ...],
  "kps_ids": [0, 1, 2, ...],  // Keypoint semantic IDs
  "n_pts": 10
}
```

### `ImageAnnotation/<category>/*.json`

Per-image annotation with keypoints and segmentation info:

```json
{
  "image": "2008_000033.jpg",
  "category": "aeroplane",
  "bndbox": {"xmin": 10, "ymin": 20, "xmax": 300, "ymax": 400},
  "kps": [[x1, y1, v1], [x2, y2, v2], ...],  // v: visibility (0/1/2)
  "segmentation": "2008_000033.png"
}
```

### `Layout/<size>/<split>_pairs.csv`

CSV files listing image pairs for each split:

```csv
src_image,trg_image,category
aeroplane/2008_000033.jpg,aeroplane/2008_000042.jpg,aeroplane
bicycle/2008_001234.jpg,bicycle/2008_001567.jpg,bicycle
...
```

---

## Usage

### Basic Loading

```python
from dataset.spair import SPairDataset
from torch.utils.data import DataLoader

# Load test split (large)
test_dataset = SPairDataset(
    root="data/SPair-71k",
    split="test",
    size="large",       # or "small" for faster experiments
    long_side=518,      # Resize long side
    normalize=True      # Apply ImageNet normalization
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
    src_img = batch['src_img']       # (B, 3, H, W)
    tgt_img = batch['tgt_img']       # (B, 3, H, W)
    src_kps = batch['src_kps']       # (B, N, 2)
    tgt_kps = batch['tgt_kps']       # (B, N, 2)
    valid_mask = batch['valid_mask']  # (B, N)
    category = batch['category']      # List[str]
    
    # Your processing...
```

### Using Small Split for Fast Experiments

```python
# Training with small split (14,160 pairs)
train_dataset = SPairDataset(
    root="data/SPair-71k",
    split="train",
    size="small",       # Much faster training
    long_side=480,
    normalize=True,
    augment=True
)

# Validation with small split
val_dataset = SPairDataset(
    root="data/SPair-71k",
    split="val",
    size="small",
    long_side=480,
    normalize=True,
    augment=False
)
```

### Loading with Segmentation

```python
# Load with segmentation masks
dataset = SPairDataset(
    root="data/SPair-71k",
    split="test",
    load_segmentation=True  # Load .png masks from Segmentation/
)

for batch in dataset:
    src_seg = batch['src_seg']  # (H, W) segmentation mask
    tgt_seg = batch['tgt_seg']  # (H, W) segmentation mask
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
- `α` is the normalized distance threshold (typically 0.05, 0.10, 0.15)
- `max(H, W)` is the maximum dimension of the bounding box

### Standard Thresholds

| Metric    | Threshold | Description                    |
|-----------|-----------|--------------------------------|
| PCK@0.05  | 5%        | Very strict (high precision)   |
| PCK@0.10  | 10%       | **Primary metric** (standard)  |
| PCK@0.15  | 15%       | Moderate tolerance             |

### Evaluation Example

```python
from dataset.spair import compute_pck

# Predict keypoints
pred_kps = model.predict(src_img, tgt_img, src_kps)

# Compute PCK
pck_results = compute_pck(
    pred_kps=pred_kps,
    gt_kps=tgt_kps,
    bbox_size=(bbox_w, bbox_h),  # Or image_size=(H, W)
    thresholds=[0.05, 0.10, 0.15]
)

print(f"PCK@0.10: {pck_results['PCK@0.10']:.4f}")
```

---

## Development Toolkit

The `devkit/` directory contains utilities for:
- Loading and visualizing annotations
- Converting between annotation formats
- Evaluation scripts
- Visualization tools

See `data/SPair-71k/devkit/README` for detailed documentation.

---

## Notes

### Split Selection
- **Large split**: Use for final evaluation and comparison with published results
- **Small split**: Use for rapid prototyping and faster experimentation

### Keypoint Visibility
Keypoints have visibility flags:
- `0`: Not labeled
- `1`: Labeled but occluded
- `2`: Labeled and visible

Always filter using the visibility mask during evaluation.

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
from pathlib import Path
assert Path("data/SPair-71k").exists(), "Dataset not found!"
```

### List available pairs
```python
import pandas as pd
pairs_df = pd.read_csv("data/SPair-71k/Layout/large/test_pairs.csv")
print(f"Found {len(pairs_df)} test pairs")
print(pairs_df.head())
```

### Memory issues
```python
# Use small split or reduce batch size
loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=False
)
```

---

## License

SPair-71k is released under the **Creative Commons Attribution-NonCommercial 4.0 International License**.

For commercial use, contact the original authors.

---

**Last Updated:** January 2026  
**Maintained by:** AML Semantic Correspondence Project
