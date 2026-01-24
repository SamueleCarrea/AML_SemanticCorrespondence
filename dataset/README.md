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
    ├── PairAnnotation/          
    │   ├── test/           
    │   ├── val/
    │   └── trn/
    │   
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
    │   │   ├── trn.txt          # (53,340 training pairs)
    │   │   ├── val.txt          # (5,384 validation pairs)
    │   │   └── test.txt         # (12,234 test pairs)
    │   └── small/               # Small subset for faster training/evaluation
    │       ├── trn.txt          # (10,652 training pairs)
    │       ├── val.txt          # (1,070 validation pairs)
    │       └── test.txt         # (2,438 test pairs)
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
  "kps_ids": [0, 1, 2, ...],
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
  "kps": [[x1, y1, v1], [x2, y2, v2], ...],
  "segmentation": "2008_000033.png"
}
```

### `Layout/<size>/<split>.txt`

**Text files** listing image pairs for each split (format: `category:src_img:trg_img` per line):

```
aeroplane:2008_000033.jpg:2008_000042.jpg
bicycle:2008_001234.jpg:2008_001567.jpg
bird:2009_000123.jpg:2009_000456.jpg
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
    normalize=True
)

# Validation with small split
val_dataset = SPairDataset(
    root="data/SPair-71k",
    split="val",
    size="small",
    long_side=480,
    normalize=True
)
```

### Loading with Segmentation

```python
# Load with segmentation masks
dataset = SPairDataset(
    root="data/SPair-71k",
    split="test",
    size="large",
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
Keypoints have visibility flags in `ImageAnnotation`:
- `0`: Not labeled
- `1`: Labeled but occluded
- `2`: Labeled and visible

Note: `PairAnnotation` only includes valid correspondences.

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

### Check split files
```python
from pathlib import Path

# Check if split files exist
layout_dir = Path("data/SPair-71k/Layout/large")
print(f"Train file exists: {(layout_dir / 'trn.txt').exists()}")
print(f"Val file exists: {(layout_dir / 'val.txt').exists()}")
print(f"Test file exists: {(layout_dir / 'test.txt').exists()}")

# Count pairs
with open(layout_dir / "test.txt") as f:
    lines = f.readlines()
    print(f"Found {len(lines)} test pairs")
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

---


# PF-WILLOW Dataset Setup

## Overview

PF-WILLOW (Proposal Flow WILLOW) is a challenging dataset for semantic correspondence evaluation, containing 900 image pairs across 4 object categories with keypoint annotations.

---

## Dataset Structure

**Actual structure in your data/ directory:**

```
data/
└── PF-WILLOW/
    ├── test_pairs.csv           # List of test image pairs
    │
    ├── car(S)/                  # Car images - Set S
    │   ├── *.png                # Image files
    │   └── *.mat                # Keypoint annotations
    │
    ├── car(M)/                  # Car images - Set M
    │   ├── *.png
    │   └── *.mat
    │
    ├── car(G)/                  # Car images - Set G
    │   ├── *.png
    │   └── *.mat
    │
    ├── duck(S)/                 # Duck images - Set S
    │   ├── *.png
    │   └── *.mat
    │
    ├── motorbike(S)/            # Motorbike images - Set S
    │   ├── *.png
    │   └── *.mat
    │
    ├── motorbike(M)/            # Motorbike images - Set M
    │   ├── *.png
    │   └── *.mat
    │
    ├── motorbike(G)/            # Motorbike images - Set G
    │   ├── *.png
    │   └── *.mat
    │
    ├── winebottle(wC)/          # Winebottle images - without Cap
    │   ├── *.png
    │   └── *.mat
    │
    ├── winebottle(woC)/         # Winebottle images - with Cap
    │   ├── *.png
    │   └── *.mat
    │
    ├── winebottle(M)/           # Winebottle images - Set M
    │   ├── *.png
    │   └── *.mat
    │
    └── .DS_Store                # macOS system file (ignore)
```

**Notes:**
- Images are in `.png` format (not `.jpg`)
- Annotations are `.mat` files in the **same directory** as images
- Multiple subsets per category (S/M/G for size variants, wC/woC for cap variants)

---

## Dataset Statistics

### Images & Pairs
- **Total Images**: ~240 across all subsets
- **Total Pairs**: 900 test pairs
- **Categories**: 4 object classes (car, duck, motorbike, winebottle)
- **Subsets**: Multiple variants per category

### Category Distribution

| Category       | Subset | Description                  | Images |
|----------------|--------|------------------------------|--------|
| car            | (S)    | Small cars                   | ~20    |
| car            | (M)    | Medium cars                  | ~20    |
| car            | (G)    | Large/General cars           | ~20    |
| duck           | (S)    | Small/Single ducks           | ~60    |
| motorbike      | (S)    | Small motorbikes             | ~20    |
| motorbike      | (M)    | Medium motorbikes            | ~20    |
| motorbike      | (G)    | Large motorbikes             | ~20    |
| winebottle     | (wC)   | With cap                     | ~20    |
| winebottle     | (woC)  | Without cap                  | ~20    |
| winebottle     | (M)    | Medium bottles               | ~20    |

### Key Characteristics
-  **Challenging poses**: Significant viewpoint and deformation changes
-  **Dense annotations**: 10 keypoints per object
-  **Multiple variants**: Different sizes and configurations
- ⚠️ **Only test split**: No training/validation data (zero-shot evaluation only)

---

## Annotation Format

### `test_pairs.csv`

CSV file listing all test pairs with their subset paths:

```csv
source_image,target_image,category
car(S)/img_001.png,car(M)/img_015.png,car
duck(S)/img_123.png,duck(S)/img_456.png,duck
motorbike(G)/img_042.png,motorbike(S)/img_089.png,motorbike
winebottle(wC)/img_010.png,winebottle(woC)/img_023.png,winebottle
...
```

### `.mat` Annotation Files (Berkeley Format)

Each image has a corresponding `.mat` file with the same name:

```
car(S)/img_001.png    → car(S)/img_001.mat
car(M)/img_015.png    → car(M)/img_015.mat
```

MATLAB `.mat` file structure:

```matlab
{
  'kps': [10 x 3]  % [x, y, visibility] for 10 keypoints
  'bbox': [x, y, w, h]  % Bounding box (optional)
}
```

**Keypoint visibility:**
- `0`: Not visible/occluded
- `1`: Visible

---

## Download Instructions

### Option 1: Official Source
```bash
# Download from official repository
cd data
wget http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip
unzip PF-dataset.zip
mv PF-dataset/PF-WILLOW ./PF-WILLOW
```

### Option 2: Alternative Mirror
```bash
git clone https://github.com/ignacio-rocco/proposalflow.git
cp -r proposalflow/datasets/pf-willow ./data/PF-WILLOW
```

---

## Usage

### Basic Loading

```python
from dataset.pf_willow import PFWillowDataset
from torch.utils.data import DataLoader

# Load test split (only split available)
test_dataset = PFWillowDataset(
    root="data/PF-WILLOW",
    long_side=518,      # Resize long side
    normalize=True      # Apply ImageNet normalization
)

# Create DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=PFWillowDataset.collate_fn
)

# Iterate
for batch in test_loader:
    src_img = batch['src_img']       # (B, 3, H, W)
    tgt_img = batch['tgt_img']       # (B, 3, H, W)
    src_kps = batch['src_kps']       # (B, 10, 2)
    tgt_kps = batch['tgt_kps']       # (B, 10, 2)
    valid_mask = batch['valid_mask']  # (B, 10)
    category = batch['category']      # List[str]
```

### Zero-Shot Evaluation

```python
# Load model trained on SPair-71k
from models.finetuner import FineTunedModel

model = FineTunedModel.load_from_checkpoint('checkpoints/best_spair.ckpt')
model.eval()

# Test on PF-WILLOW (zero-shot)
pf_dataset = PFWillowDataset(root="data/PF-WILLOW")
results = evaluate_pck(model, pf_dataset, thresholds=[0.05, 0.10, 0.15])

print(f"PF-WILLOW PCK@0.10: {results['PCK@0.10']:.2f}%")
```

---

## Evaluation Metrics

Same as SPair-71k: **Percentage of Correct Keypoints (PCK)**

### PCK@α Definition

```
||pred_kp - gt_kp||₂ ≤ α × max(H, W)
```

### Standard Thresholds

| Metric    | Threshold | Description                    |
|-----------|-----------|--------------------------------|
| PCK@0.05  | 5%        | Very strict                    |
| PCK@0.10  | 10%       | **Primary metric**             |
| PCK@0.15  | 15%       | Moderate tolerance             |

---

## Comparison with SPair-71k

| Feature           | SPair-71k        | PF-WILLOW       |
|-------------------|------------------|-----------------|
| **Images**        | 1,800            | ~240            |
| **Pairs**         | 70,958           | 900             |
| **Categories**    | 18               | 4 (10 subsets)  |
| **Keypoints**     | Variable (6-20)  | Fixed (10)      |
| **Image Format**  | .jpg             | .png            |
| **Annotation**    | JSON             | MATLAB .mat     |
| **Splits**        | Train/Val/Test   | Test only       |
| **Difficulty**    | Medium           | High            |
| **Use Case**      | Training         | Zero-shot eval  |

---

## Expected Performance Gap

Typical performance drops when transferring from SPair-71k to PF-WILLOW:

| Method              | SPair-71k PCK@0.10 | PF-WILLOW PCK@0.10 | Gap    |
|---------------------|--------------------|--------------------|--------|
| Frozen features     | ~45%               | ~35%               | -10%   |
| Fine-tuned          | ~55%               | ~42%               | -13%   |
| + Soft-argmax       | ~58%               | ~45%               | -13%   |

**Why the gap?**
- More extreme pose variations
- Different object categories
- Cross-subset matching (e.g., car(S) → car(G))
- Only 4 classes vs 18 (less category overlap)
- No training data for adaptation

---

## Visualization

```python
from utils.visualization import visualize_correspondences

# Visualize predictions
for batch in test_loader:
    pred_kps = model.predict(batch)
    
    fig = visualize_correspondences(
        src_img=batch['src_img'][0],
        tgt_img=batch['tgt_img'][0],
        src_kps=batch['src_kps'][0],
        tgt_kps=batch['tgt_kps'][0],
        pred_kps=pred_kps[0],
        category=batch['category'][0]
    )
    fig.savefig(f"results/pf_willow_{batch['category'][0]}.png")
```

---

## Citation

If you use PF-WILLOW in your research, please cite:

```bibtex
@inproceedings{ham2017proposal,
  title={Proposal Flow},
  author={Ham, Bumsub and Cho, Minsu and Schmid, Cordelia and Ponce, Jean},
  booktitle={CVPR},
  year={2017}
}
```

---

## Troubleshooting

### Dataset not found
```python
from pathlib import Path
assert Path("data/PF-WILLOW").exists(), "Dataset not found!"
assert Path("data/PF-WILLOW/test_pairs.csv").exists(), "test_pairs.csv missing!"
```

### MATLAB file reading issues
```bash
# Install scipy for .mat file support
pip install scipy
```

```python
import scipy.io as sio
mat_data = sio.loadmat('car(S)/image_001.mat')
kps = mat_data['kps']  # (10, 3) array
```

### Check dataset integrity
```python
from pathlib import Path

# Check all subsets
subsets = [
    'car(S)', 'car(M)', 'car(G)',
    'duck(S)',
    'motorbike(S)', 'motorbike(M)', 'motorbike(G)',
    'winebottle(wC)', 'winebottle(woC)', 'winebottle(M)'
]

for subset in subsets:
    subset_dir = Path(f"data/PF-WILLOW/{subset}")
    if subset_dir.exists():
        n_png = len(list(subset_dir.glob('*.png')))
        n_mat = len(list(subset_dir.glob('*.mat')))
        print(f"{subset:20s} - PNG: {n_png:3d}, MAT: {n_mat:3d}")
    else:
        print(f"{subset:20s} - MISSING!")
```

### Ignore system files
The `.DS_Store` file is a macOS system file and can be safely ignored.

---

## License

PF-WILLOW is released for academic research purposes.  
For commercial use, contact the original authors.

---

**Last Updated:** January 2026  
**Maintained by:** AML Semantic Correspondence Project


