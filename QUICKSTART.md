# Quick Start Guide

## Running the Baseline

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pillow scipy scikit-image opencv-python wandb pyyaml tqdm timm scikit-learn
```

Or use the setup script:
```bash
bash scripts/setup.sh
```

### 2. Test the Implementation

Run the test script to verify everything is working:
```bash
python scripts/test_baseline.py
```

Expected output:
```
Testing Semantic Correspondence Baseline...
============================================================

1. Testing imports...
   ✓ All imports successful

2. Testing seed setting...
   ✓ Seed setting successful

3. Testing dataset loader...
   ✓ Dataset loader working (sample keys: [...])

4. Testing feature extraction...
   ✓ Feature extraction working (shape: [1, 384, 14, 14])

5. Testing matching...
   ✓ Matching working (X matches found)

6. Testing PCK metric...
   ✓ PCK metric working (PCK@0.1: XX.XX%)

7. Testing logger...
   ✓ Logger working

============================================================
All tests passed! ✓
```

### 3. Run Evaluation

#### With dummy data (no dataset required):
```bash
python eval.py --data_dir data/dummy --no-wandb
```

#### With SPair-71k dataset:
```bash
# Download and extract SPair-71k to data/SPair-71k/
python eval.py
```

#### With DINOv2:
```bash
python eval.py --config configs/dinov2.yaml
```

### Expected Evaluation Output

```
Using device: cuda
Loading dataset...
Loaded 1000 image pairs
Initializing feature extractor...
Starting evaluation...
Evaluating: 100%|████████████| 1000/1000 [10:00<00:00,  1.67it/s]

==================================================
Evaluation Results:
==================================================
Evaluated 1000 samples
  pck@0.05: 45.67%
  pck@0.1: 62.34%
  pck@0.15: 71.23%
```

## Command Line Options

```bash
python eval.py [OPTIONS]

Options:
  --config PATH       Path to config file (default: configs/default.yaml)
  --data_dir PATH     Dataset directory (overrides config)
  --device DEVICE     Device to use: cuda/cpu (overrides config)
  --no-wandb          Disable WandB logging
```

## Configuration

Edit `configs/default.yaml` or create a new config file:

```yaml
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

## Troubleshooting

### Out of Memory
- Reduce `dataset.img_size` in config
- Use CPU: `--device cpu`
- Process fewer images at once

### Models Not Downloading
- Check internet connection
- Models are downloaded automatically on first run
- DINO models come from timm
- DINOv2 models from torch.hub (Facebook Research)

### No Dataset Available
- Use dummy data: `--data_dir data/dummy --no-wandb`
- Download SPair-71k from https://cvlab.postech.ac.kr/research/SPair-71k/

## Next Steps

1. **Experiment with different models**: Try DINOv2 variants (small, base, large)
2. **Tune matching**: Enable RANSAC filtering
3. **Add new features**: Extend with diffusion model features
4. **Optimize performance**: Profile and optimize bottlenecks
5. **Add training**: Implement learnable matching layers

## Project Structure

```
├── configs/           # YAML configuration files
├── data/             # Dataset directory
├── src/
│   ├── datasets/     # Dataset loaders (SPair-71k)
│   ├── features/     # Feature extractors (DINO/DINOv2)
│   ├── matching/     # Matching algorithms (NN, mutual NN, RANSAC)
│   └── metrics/      # Evaluation metrics (PCK)
├── utils/            # Utilities (seed, logger)
├── scripts/          # Helper scripts
└── eval.py           # Main evaluation script
```
