# Semantic Correspondence with DINO/DINOv2

A PyTorch implementation of a training-free baseline for semantic correspondence using DINO and DINOv2 features.

## Overview

This repository provides a clean, modular baseline for establishing semantic correspondences between image pairs. The implementation uses self-supervised vision transformers (DINO/DINOv2) to extract dense features and matches them using nearest neighbor search.

## Features

- **Feature Extraction**: DINO and DINOv2 vision transformers
- **Matching**: Nearest neighbor and mutual nearest neighbor matching
- **Optional RANSAC**: Geometric verification of matches
- **Evaluation**: PCK (Percentage of Correct Keypoints) metric at multiple thresholds
- **Dataset Support**: SPair-71k dataset loader
- **Logging**: WandB integration for experiment tracking
- **Reproducibility**: Seed setting for deterministic results

## Project Structure

```
.
├── configs/                    # Configuration files
│   ├── default.yaml           # Default DINO configuration
│   └── dinov2.yaml            # DINOv2 configuration
├── data/                       # Dataset directory
│   └── SPair-71k/             # SPair-71k dataset (download separately)
├── src/
│   ├── datasets/              # Dataset loaders
│   │   └── spair.py          # SPair-71k loader
│   ├── features/              # Feature extraction
│   │   └── dino.py           # DINO/DINOv2 extractors
│   ├── matching/              # Matching algorithms
│   │   └── matcher.py        # NN, mutual NN, RANSAC
│   └── metrics/               # Evaluation metrics
│       └── pck.py            # PCK computation
├── utils/                      # Utility functions
│   ├── seed.py               # Seed setting
│   └── logger.py             # WandB logger
├── eval.py                     # Main evaluation script
└── requirements.txt            # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SamueleCarrea/AML_SemanticCorrespondence.git
cd AML_SemanticCorrespondence
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### SPair-71k

1. Download the SPair-71k dataset from [the official website](https://cvlab.postech.ac.kr/research/SPair-71k/)

2. Extract the dataset to the `data/` directory:
```bash
mkdir -p data
# Extract SPair-71k to data/SPair-71k/
```

Expected directory structure:
```
data/SPair-71k/
├── ImageAnnotation/
├── JPEGImages/
└── PairAnnotation/
```

**Note**: The code includes a fallback to dummy data if the dataset is not available, allowing you to test the pipeline without downloading the full dataset.

## Usage

### Basic Evaluation

Run evaluation with default DINO configuration:
```bash
python eval.py
```

### Using DINOv2

Run evaluation with DINOv2:
```bash
python eval.py --config configs/dinov2.yaml
```

### Command Line Options

```bash
python eval.py \
    --config configs/default.yaml \     # Configuration file
    --data_dir data/SPair-71k \         # Dataset directory
    --device cuda \                      # Device (cuda/cpu)
    --no-wandb                          # Disable WandB logging
```

### Custom Configuration

Create your own configuration file based on `configs/default.yaml`:

```yaml
# my_config.yaml
dataset:
  root_dir: "data/SPair-71k"
  img_size: 224

features:
  model_type: "dino"  # or "dinov2"
  model_name: "vit_small_patch16_224.dino"

matching:
  type: "mutual_nn"  # or "nn"
  use_ransac: false

evaluation:
  alphas: [0.05, 0.1, 0.15]

logging:
  use_wandb: true
  project: "semantic-correspondence"

seed: 42
device: "cuda"
```

Run with custom config:
```bash
python eval.py --config my_config.yaml
```

## Available Models

### DINO (via timm)
- `vit_small_patch16_224.dino`
- `vit_small_patch8_224.dino`
- `vit_base_patch16_224.dino`
- `vit_base_patch8_224.dino`

### DINOv2 (via torch.hub or timm)
- `small` (dinov2_vits14)
- `base` (dinov2_vitb14)
- `large` (dinov2_vitl14)
- `giant` (dinov2_vitg14)

## Matching Algorithms

1. **Nearest Neighbor (NN)**: Find the closest match in target for each source point
2. **Mutual Nearest Neighbor**: Keep only bidirectionally consistent matches
3. **RANSAC** (optional): Filter matches using geometric verification

## Evaluation Metrics

**PCK (Percentage of Correct Keypoints)**: Measures the percentage of predicted keypoints within a threshold distance from ground truth.

Default thresholds (α):
- PCK@0.05: 5% of image size
- PCK@0.1: 10% of image size
- PCK@0.15: 15% of image size

## Logging with WandB

The code integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking:

1. Install and login to WandB:
```bash
pip install wandb
wandb login
```

2. Run evaluation (WandB enabled by default):
```bash
python eval.py
```

3. Disable WandB:
```bash
python eval.py --no-wandb
```

## Testing Without Dataset

The code can run without the actual SPair-71k dataset by using dummy data:

```bash
python eval.py --data_dir data/dummy --no-wandb
```

This is useful for:
- Testing the pipeline
- Debugging code changes
- CI/CD integration

## Development

### Adding New Features

1. **New Feature Extractor**: Add to `src/features/`
2. **New Matching Algorithm**: Add to `src/matching/`
3. **New Metric**: Add to `src/metrics/`
4. **New Dataset**: Add to `src/datasets/`

### Code Structure

The codebase follows a modular design:
- Each component is independent and can be used separately
- Configuration-driven: All hyperparameters in YAML files
- Logging: Integrated WandB support
- Reproducibility: Seed setting utilities

## Citation

If you use this code in your research, please cite:

```bibtex
@software{semantic_correspondence_baseline,
  author = {Samuele Carrea},
  title = {Semantic Correspondence Baseline with DINO/DINOv2},
  year = {2024},
  url = {https://github.com/SamueleCarrea/AML_SemanticCorrespondence}
}
```

## References

- **DINO**: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
- **DINOv2**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- **SPair-71k**: [SPair-71k: A Large-scale Benchmark for Semantic Correspondence](https://arxiv.org/abs/1908.10543)

## License

This project is licensed under the MIT License.

## Acknowledgments

- DINO and DINOv2 models from Facebook AI Research
- SPair-71k dataset from POSTECH CVLab
- timm library for vision models