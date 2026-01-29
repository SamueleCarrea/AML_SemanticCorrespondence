# Semantic Correspondence with Visual Foundation Models

This project implements semantic correspondence methods using pretrained visual foundation models (DINOv2, DINOv3, SAM) on the SPair-71k benchmark dataset.

##  Project Overview

**Semantic correspondence** is the task of finding pixel-level matches between semantically similar parts of objects across different images. For example, given a point on the left eye of a dog in one image, the goal is to find the corresponding left eye in another image of a dog, or even semantically related objects like wolves or cartoon dogs.

### Key Challenges
- Objects appear in different viewpoints, scales, and contexts
- Images may come from different domains (photos vs. paintings)
- Models must distinguish semantically similar but geometrically different parts (e.g., left vs. right paw)

##  Project Goals

The project is divided into four main stages:

1. **Training-free baseline**: Use frozen features from pretrained backbones for correspondence
2. **Light fine-tuning**: Adapt the last layers of the backbone with keypoint supervision
3. **Better prediction rule**: Replace argmax with window soft-argmax for sub-pixel refinement
4. **Extension**: Cross-dataset generalization evaluation

##  Project Structure

```
AML_SemanticCorrespondence/
├── dataset/                    # Dataset loading and preprocessing
│   ├── __init__.py            # Package initialization
│   ├── README.md              # Dataset documentation
│   ├── spair.py               # SPair-71k dataset loader
│   └── willow.py              # PF-Willow dataset loader
├── models/                     # Model implementations
│   ├── __init__.py            # Package initialization
│   ├── backbones.py           # Individual backbone implementations (DINOv2, DINOv3, SAM)
│   ├── unified_backbone.py    # Unified interface for all backbones
│   ├── matcher.py             # Correspondence matching logic (argmax, soft-argmax)
│   ├── finetuner.py           # Fine-tuning logic for last layers
│   ├── evaluator.py           # Model evaluation and metrics computation
│   ├── loss.py                # Loss functions for training
│   └── config.py              # Model configuration classes
├── utils/                      # Utility functions
│   ├── __init__.py            # Package initialization
│   └── pck.py                 # PCK metric computation utilities
├── notebooks/                  # Jupyter notebooks for experiments
│   ├── train.ipynb            # Training pipeline and experiments
│   └── eval.ipynb             # Evaluation and analysis
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

##  Dataset Module

The `dataset/` folder contains data loading utilities for semantic correspondence benchmarks:

### Files Description

- **`__init__.py`**: Package initialization file that exports dataset classes and utilities
- **`README.md`**: Detailed documentation about dataset structure, formats, and usage
- **`spair.py`**: SPair-71k dataset implementation
  - Loads image pairs with annotated keypoints
  - Provides 18 object categories (e.g., dog, cat, car, bicycle)
  - Includes ~70,000 image pairs with semantic keypoint annotations
  - Supports train/validation/test splits
  
- **`willow.py`**: PF-Willow dataset implementation
  - Alternative benchmark for semantic correspondence
  - Provides 10 object categories
  - Smaller dataset focused on common objects

### Dataset Format

Each dataset provides image pairs with:
- **Source image**: Image with annotated keypoints
- **Target image**: Image where corresponding keypoints need to be predicted
- **Keypoint annotations**: Semantic part locations (e.g., "left eye", "front wheel")
- **Metadata**: Object category, viewpoint, scale information

### Usage Example

```python
from dataset import SPairDataset
from dataset import PFWillowDataset

# Load SPair-71k dataset for training
train_dataset_full = SPairDataset(
  root=SPAIR_ROOT,
  split='train', 
  size='large', 
  long_side=IMG_SIZE,
  normalize=True, 
  load_segmentation=False
)

# Load PF-Willow dataset for cross-dataset evaluation
test_dataset = PFWillowDataset(
  root=DATASET_ROOT,
  long_side=IMG_SIZE,
  normalize=True
)

# Define a sample
sample = {
    "src_img": src_tensor,              # Source image tensor
    "tgt_img": tgt_tensor,              # Target image tensor
    "src_kps": src_kps,                 # Source keypoints
    "tgt_kps": tgt_kps,                 # Target keypoints
    "valid_mask": valid_mask,           # Binary mask indicating valid keypoints
    "category": category,               # Object category string (e.g., "cat", "dog")
    "pair_id": pair_id,                 # Unique identifier for the image pair
    "src_scale": torch.tensor(src_scale, dtype=torch.float32),  # Scale factor for source image
    "tgt_scale": torch.tensor(tgt_scale, dtype=torch.float32),  # Scale factor for target image
    "src_orig_size": torch.tensor(src_orig_size, dtype=torch.int64),  # Original source size
    "tgt_orig_size": torch.tensor(tgt_orig_size, dtype=torch.int64)  # Original target size
}
```

##  Models Module

The `models/` folder contains the core implementation of backbone models, matching algorithms, and training components:

### Files Description

- **`__init__.py`**: Package initialization
- **`backbones.py`**: Individual implementations of vision foundation models
  - DINOv2 feature extractor
  - DINOv3 feature extractor
  - SAM (Segment Anything Model) feature extractor
  - Handles model loading, preprocessing, and feature extraction

- **`unified_backbone.py`**: Unified interface for all backbones
  - Handles backbone selection and initialization
  - Simplifies switching between models

- **`matcher.py`**: Correspondence matching algorithms
  - Argmax-based matching 
  - Window soft-argmax matching 
  - Cosine similarity computation
  - Sub-pixel refinement logic

- **`finetuner.py`**: Fine-tuning implementation
  - Layer unfreezing logic
  - Training loop for last N layers
  - Gradient management and optimization

- **`evaluator.py`**: Evaluation pipeline
  - PCK@T metric computation
  - Per-keypoint and per-image evaluation
  - Results aggregation and reporting

- **`loss.py`**: Loss functions
  - Keypoint correspondence loss
  - Distance-based losses
  - Custom loss implementations for semantic matching

- **`config.py`**: Configuration classes
  - Backbones configurations

## Utils Module

The `utils/` folder contains utility functions for metrics and evaluation:

### Files Description

- **`__init__.py`**: Package initialization
- **`pck.py`**: PCK (Percentage of Correct Keypoints) metric implementation
  - Computes PCK@T for multiple thresholds simultaneously
  - **Dual normalization support**: bounding box or image dimensions
  - Returns per-threshold results as tuples: `(valid_mask, accuracy)`
  - `valid_mask`: (N,) binary tensor indicating which keypoints are correct
  - `accuracy`: float, mean accuracy for this threshold


### Usage Example

```python
from models import UnifiedBackbone
from models import CorrespondenceMatcher
from models import UnifiedEvaluator
from utils import compute_pck

# Definition of backbones, finetuning, dataset and prediction method
backbone_choice = 'dinov2'  # 'dinov2', 'dinov3', 'sam'
finetune_choice = False     # True, False
soft_argmax_choice = False  # True, False
dataset_choice = 'spair'    # 'spair', 'pfwillow'


# Initialize backbone
backbone = UnifiedBackbone(
  backbone_choice=backbone_choice,
  finetune_choice=finetune_choice, 
  checkpoint_dir=CHECKPOINT_DIR,
  device=device
)

# Initialize matcher
matcher = CorrespondenceMatcher(
  backbone=backbone,
  use_soft_argmax=soft_argmax_choice
)

# Initialize evaluator
evaluator = UnifiedEvaluator(
  dataloader=test_loader,
  device=device,
  thresholds=[0.05, 0.10, 0.15, 0.20]
)

# Extract features and match
src_feat = self.backbone.extract_features(src_img)[0]
tgt_feat = self.backbone.extract_features(tgt_img)[0]

tgt_kps_pred = matcher.match(src_img, tgt_img, src_kps_valid)

# Compute PCK across multiple thresholds
pck_scores = compute_pck(
  tgt_kps_pred_orig, 
  tgt_kps_gt_orig, 
  bbox_size=bbox_size,              
  image_size=(H_orig, W_orig),
  thresholds=self.thresholds
)
```

##  Notebooks Module

The `notebooks/` folder contains Jupyter notebooks for interactive experimentation and analysis:

### Files Description

- **`eval.ipynb`**: Evaluation and analysis
  - **Configuration** : Select backbone, finetuning, soft-argmax, dataset
  - Model evaluation on SPair-71k/PF-Willow test set
  - PCK metric computation (per-keypoint and per-image)
  - Results visualization
  - Results saved as JSON files


- **`train.ipynb`**: Training pipeline and experiments
  - **Training modes**: fresh, resume, continue
  - Baseline model training (frozen features)
  - Fine-tuning experiments with different layer configurations
  - Hyperparameter exploration
  - Training visualization and logging
  - Model checkpoint management (saves optimizer, scheduler, scaler state)

##  Methodology

### 1. Training-Free Baseline
- Extract dense features from frozen pretrained backbones (DINOv2, DINOv3, SAM)
- Compute cosine similarity between source keypoint features and all target patches
- Select the location with highest similarity (argmax) as the predicted match
- **Evaluation**: Test on SPair-71k test set
- **Implementation**: `models/backbones.py`, `models/matcher.py`, `models/evaluator.py`, `utils/pck.py`
- **Notebook**: `notebooks/eval.ipynb`
- **Eval configuration (Cell 2)**:
  - `backbone_choice`: `'dinov2'` | `'dinov3'` | `'sam'` (run one at a time)
  - `finetune_choice`: `False`
  - `soft_argmax_choice`: `False`
  - `dataset_choice`: `'spair'`

### 2. Light Fine-tuning
- Unfreeze and fine-tune the last N layers of each backbone (DINOv2, DINOv3, SAM)
- Train with keypoint supervision from SPair-71k training set
- Compare performance across different numbers of unfrozen layers
- **Evaluation**: Test on SPair-71k validation set
- **Implementation**: `models/finetuner.py`, `models/loss.py`
- **Notebook**: `notebooks/train.ipynb`
- **Train configuration (Cell 3 – `FinetuneConfig`)**:
  - **Backbone** (run separate experiments):
    - DINOv2: `backbone_name = 'dinov2_vitb14'`
    - DINOv3: `backbone_name = 'dinov3_vitb16'`
    - SAM: `backbone_name = 'sam_vit_b'`
  - **Unfreezing depth**: `num_layers_to_finetune = 1..N` (e.g. `2`)
  - **Core params**: `num_epochs`, `learning_rate`, `weight_decay`, `warmup_epochs`, `loss_type`, `temperature`
  - **Data**: `batch_size`, `num_workers`, `max_train_pairs`, `max_val_pairs`

### 3. Window Soft-Argmax Prediction
- Find peak location using standard argmax
- Apply soft-argmax within a small window around the peak
- Enables sub-pixel refinement and robustness to noisy similarity maps
- **Evaluation**: Test on SPair-71k validation set
- **Implementation**: `models/matcher.py`
- **Notebook**: `notebooks/eval.ipynb`
- **Eval configuration (Cell 2)**:
  - `backbone_choice`: `'dinov2'` | `'dinov3'` | `'sam'` (run one at a time)
  - `finetune_choice`: `False`
  - `soft_argmax_choice`: `True`
  - `dataset_choice`: `'spair'`

### 4. Cross-Dataset Generalization (Extension)
**Goal**: Evaluate how well the three fine-tuned backbones generalize to a different dataset

**Experimental Setup**:
- **Training**: Fine-tune DINOv2, DINOv3, and SAM on SPair-71k
- **Evaluation**: Test all three fine-tuned models on PF-Willow dataset
- **Comparison**: Analyze performance differences between:
  - Cross-domain evaluation (SPair → Willow)
- **Implementation**: `models/evaluator.py`, `utils/pck.py`, `dataset/willow.py`
- **Notebook**: `notebooks/eval.ipynb`
- **Eval configuration (Cell 2)**:
  - `backbone_choice`: `'dinov2'` | `'dinov3'` | `'sam'` (run one at a time)
  - `finetune_choice`: `True` (uses fine-tuned checkpoints in `CHECKPOINT_DIR`)
  - `soft_argmax_choice`: `False` (set `True` only if you want soft-argmax evaluation)
  - `dataset_choice`: `'pfwillow'`


## Evaluation Metrics

**PCK@T (Percentage of Correct Keypoints)**: Measures the percentage of predicted keypoints within a normalized distance T from ground truth.

**Formula**: A keypoint is correct if `||pred_kp - gt_kp||_2 <= T * max(H, W)`

- Multiple thresholds evaluated: 0.05, 0.10, 0.15, 0.20
- **Two evaluation modes**:
  - **Per-keypoint**: Individual keypoint accuracy (default, aligns with SPair-71k benchmark)
  - **Per-image**: Average accuracy across all keypoints in an image
- **Normalization strategies**:
  - Primary: Bounding box dimensions (when available)
  - Fallback: Original image dimensions (preserves aspect ratio, not resized)
- Results reported with mean and standard deviation
- Analysis across object categories and difficulty levels

**Evaluation Protocol**:
- **In-domain**: SPair-71k train → SPair-71k validation
- **Cross-domain** (Extension): SPair-71k train → PF-Willow test
- **Backbone comparison**: DINOv2, DINOv3, SAM on both evaluation sets

