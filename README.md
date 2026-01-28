# Semantic Correspondence with Visual Foundation Models

This project implements semantic correspondence methods using pretrained visual foundation models (DINOv2, DINOv3, SAM) on the SPair-71k benchmark dataset.

## 📋 Project Overview

**Semantic correspondence** is the task of finding pixel-level matches between semantically similar parts of objects across different images. For example, given a point on the left eye of a dog in one image, the goal is to find the corresponding left eye in another image of a dog, or even semantically related objects like wolves or cartoon dogs.

### Key Challenges
- Objects appear in different viewpoints, scales, and contexts
- Images may come from different domains (photos vs. paintings)
- Models must distinguish semantically similar but geometrically different parts (e.g., left vs. right paw)

## 🎯 Project Goals

The project is divided into four main stages:

1. **Training-free baseline**: Use frozen features from pretrained backbones for correspondence
2. **Light fine-tuning**: Adapt the last layers of the backbone with keypoint supervision
3. **Better prediction rule**: Replace argmax with window soft-argmax for sub-pixel refinement
4. **Extension**: Cross-dataset generalization evaluation

## 📁 Project Structure

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

## 📦 Dataset Module

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
from dataset.spair import SPairDataset
from dataset.willow import WillowDataset

# Load SPair-71k dataset for training
train_dataset = SPairDataset(
    root='path/to/SPair-71k',
    split='train',
    category='dog'
)

# Load PF-Willow dataset for cross-dataset evaluation
eval_dataset = WillowDataset(
    root='path/to/PF-Willow',
    split='test',
    category='dog'
)

# Get a sample
sample = train_dataset[0]
src_img = sample['src_img']      # Source image
tgt_img = sample['tgt_img']      # Target image
src_kps = sample['src_kps']      # Source keypoints (N, 2)
tgt_kps = sample['tgt_kps']      # Target keypoints (N, 2)
```

## 🤖 Models Module

The `models/` folder contains the core implementation of backbone models, matching algorithms, and training components:

### Files Description

- **`__init__.py`**: Package initialization
- **`backbones.py`**: Individual implementations of vision foundation models
  - DINOv2 feature extractor
  - DINOv3 feature extractor
  - SAM (Segment Anything Model) feature extractor
  - Handles model loading, preprocessing, and feature extraction

- **`unified_backbone.py`**: Unified interface for all backbones
  - Provides consistent API across different models
  - Handles backbone selection and initialization
  - Simplifies switching between models

- **`matcher.py`**: Correspondence matching algorithms
  - Argmax-based matching (baseline)
  - Window soft-argmax matching (improved)
  - Cosine similarity computation
  - Sub-pixel refinement logic

- **`finetuner.py`**: Fine-tuning implementation
  - Layer unfreezing logic
  - Training loop for last N layers
  - Gradient management and optimization

- **`evaluator.py`**: Evaluation pipeline
  - PCK@T metric computation
  - Per-keypoint and per-image evaluation
  - Cross-dataset evaluation support
  - Results aggregation and reporting

- **`loss.py`**: Loss functions
  - Keypoint correspondence loss
  - Distance-based losses
  - Custom loss implementations for semantic matching

- **`config.py`**: Configuration classes
  - Model hyperparameters
  - Training configurations
  - Evaluation settings
  - Dataset-specific parameters

### Usage Example

```python
from models.unified_backbone import UnifiedBackbone
from models.matcher import SemanticMatcher
from models.evaluator import Evaluator

# Initialize backbone
backbone = UnifiedBackbone(
    model_name='dinov2',
    num_layers_to_finetune=2
)

# Initialize matcher
matcher = SemanticMatcher(
    method='soft_argmax',
    window_size=5
)

# Extract features and match
src_features = backbone.extract_features(src_img)
tgt_features = backbone.extract_features(tgt_img)
predictions = matcher.match(src_features, tgt_features, src_kps)

# Evaluate
evaluator = Evaluator(thresholds=[0.05, 0.1, 0.15, 0.2])
pck_scores = evaluator.compute_pck(predictions, tgt_kps)
```

## 🛠️ Utils Module

The `utils/` folder contains utility functions for metrics and evaluation:

### Files Description

- **`__init__.py`**: Package initialization
- **`pck.py`**: PCK (Percentage of Correct Keypoints) metric implementation
  - Computes PCK@T for multiple thresholds simultaneously
  - Handles normalized distance calculation using bounding box or image size
  - Returns per-threshold results in a dictionary format
  - Supports batch evaluation across multiple image pairs

### Usage Example

```python
from utils.pck import compute_pck

# Compute PCK across multiple thresholds
pck_results = compute_pck(
    pred_kps=predicted_kps,        # (N, 2) predicted keypoints
    gt_kps=ground_truth_kps,       # (N, 2) ground truth keypoints
    bbox_size=(height, width),     # Bounding box for normalization
    thresholds=[0.05, 0.10, 0.15, 0.20]
)

# Access results
for metric, (correct_mask, accuracy) in pck_results.items():
    print(f"{metric}: {accuracy:.4f}")
    # correct_mask: (N,) binary tensor indicating which keypoints are correct
    # accuracy: float, mean accuracy for this threshold
```

## 📓 Notebooks Module

The `notebooks/` folder contains Jupyter notebooks for interactive experimentation and analysis:

### Files Description

- **`train.ipynb`**: Training pipeline and experiments
  - Step-by-step training workflow
  - Baseline model training (frozen features)
  - Fine-tuning experiments with different layer configurations
  - Hyperparameter exploration
  - Training visualization and logging
  - Model checkpoint management
  - Comparative analysis across backbones (DINOv2, DINOv3, SAM)

- **`eval.ipynb`**: Evaluation and analysis
  - Model evaluation on SPair-71k validation set
  - Cross-dataset evaluation on PF-Willow
  - PCK metric computation and visualization
  - Per-category performance breakdown
  - Qualitative results visualization
  - Error analysis and failure case inspection
  - Comparison of argmax vs. soft-argmax matching
  - Cross-backbone performance comparison

### Usage

```bash
# Start Jupyter notebook server
jupyter notebook

# Navigate to notebooks/ folder and open desired notebook
```

## 🔬 Methodology

### 1. Training-Free Baseline
- Extract dense features from frozen pretrained backbones (DINOv2, DINOv3, SAM)
- Compute cosine similarity between source keypoint features and all target patches
- Select the location with highest similarity (argmax) as the predicted match
- **Evaluation**: Test on SPair-71k validation set
- **Implementation**: `models/backbones.py`, `models/matcher.py`
- **Notebook**: `notebooks/eval.ipynb`

### 2. Light Fine-tuning
- Unfreeze and fine-tune the last N layers of each backbone (DINOv2, DINOv3, SAM)
- Train with keypoint supervision from SPair-71k training set
- Compare performance across different numbers of unfrozen layers
- **Evaluation**: Test on SPair-71k validation set
- **Implementation**: `models/finetuner.py`, `models/loss.py`
- **Notebook**: `notebooks/train.ipynb`

### 3. Window Soft-Argmax Prediction
- Find peak location using standard argmax
- Apply soft-argmax within a small window around the peak
- Enables sub-pixel refinement and robustness to noisy similarity maps
- **Evaluation**: Test on SPair-71k validation set
- **Implementation**: `models/matcher.py`
- **Notebook**: `notebooks/eval.ipynb`

### 4. Cross-Dataset Generalization (Extension)
**Goal**: Evaluate how well the three fine-tuned backbones generalize to a different dataset

**Experimental Setup**:
- **Training**: Fine-tune DINOv2, DINOv3, and SAM on SPair-71k
- **Evaluation**: Test all three fine-tuned models on PF-Willow dataset
- **Comparison**: Analyze performance differences between:
  - In-domain evaluation (SPair → SPair)
  - Cross-domain evaluation (SPair → Willow)
- **Implementation**: `models/evaluator.py`, `utils/pck.py`
- **Notebook**: `notebooks/eval.ipynb`

**Research Questions**:
1. Which backbone generalizes best to PF-Willow after SPair-71k fine-tuning?
2. How much does performance degrade when transferring across datasets?
3. Are certain object categories more robust to domain shift?
4. Does fine-tuning improve or hurt cross-dataset generalization compared to frozen features?

**Analysis Dimensions**:
- Per-backbone comparison (DINOv2 vs DINOv3 vs SAM)
- Per-category analysis (which objects transfer better)
- Performance degradation: (SPair PCK) - (Willow PCK)
- Comparison with training-free baseline on both datasets

**Expected Insights**:
- Understand which visual foundation model learns more generalizable semantic features
- Identify potential overfitting to SPair-71k characteristics
- Guide model selection for real-world deployment scenarios

## 📊 Evaluation Metrics

**PCK@T (Percentage of Correct Keypoints)**: Measures the percentage of predicted keypoints within a normalized distance T from ground truth.

**Formula**: A keypoint is correct if `||pred_kp - gt_kp||_2 <= T * max(H, W)`

- Multiple thresholds evaluated: 0.05, 0.10, 0.15, 0.20
- Normalization by maximum dimension of bounding box or image
- Results reported per keypoint and per image
- Analysis across object categories and difficulty levels

**Evaluation Protocol**:
- **In-domain**: SPair-71k train → SPair-71k validation
- **Cross-domain** (Extension): SPair-71k train → PF-Willow test
- **Backbone comparison**: DINOv2, DINOv3, SAM on both evaluation sets

**Implementation**: `utils/pck.py`, `models/evaluator.py`

## 🚀 Getting Started

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/AML_SemanticCorrespondence.git
cd AML_SemanticCorrespondence
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: This project is optimized for Google Colab. Some dependencies (PyTorch, NumPy) are commented out in `requirements.txt` as they come pre-installed in Colab environments.

### Key Dependencies

- **PyTorch Stack**: torch>=2.0.0, torchvision>=0.15.0
- **Vision Models**: timm>=0.9.0, torchmetrics>=1.0.0
- **Foundation Models**:
  - DINOv2 (loaded via `torch.hub`)
  - DINOv3 (via timm or torch.hub)
  - SAM (optional, install separately if needed)
- **Data Processing**: PIL, OpenCV, scikit-image
- **Utilities**: einops, tqdm, omegaconf
- **Visualization**: matplotlib, seaborn, plotly
- **Experiment Tracking**: wandb, tensorboard

### Running the Project

**Training**:
```bash
# Open train.ipynb in Jupyter or Colab
jupyter notebook notebooks/train.ipynb
```

**Evaluation**:
```bash
# Open eval.ipynb in Jupyter or Colab
jupyter notebook notebooks/eval.ipynb
```

### Google Colab Setup

```python
# Mount Google Drive (optional, for dataset storage)
from google.colab import drive
drive.mount('/content/drive')

# Install requirements
!pip install -r requirements.txt

# Import modules
from dataset.spair import SPairDataset
from models.unified_backbone import UnifiedBackbone
```

