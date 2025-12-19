"""
Example usage of the semantic correspondence baseline.

This script demonstrates how to use the baseline components
without running the full evaluation.
"""

# Example 1: Loading and using the dataset
print("=" * 60)
print("Example 1: Dataset Loading")
print("=" * 60)
print("""
from src.datasets.spair import SPairDataset

# Create dataset
dataset = SPairDataset(
    root_dir="data/SPair-71k",
    split="test",
    category="cat",
    img_size=224
)

# Get a sample
sample = dataset[0]
src_img = sample['src_img']  # Source image [3, 224, 224]
trg_img = sample['trg_img']  # Target image [3, 224, 224]
src_kps = sample['src_kps']  # Source keypoints [N, 2]
trg_kps = sample['trg_kps']  # Target keypoints [N, 2]
""")

# Example 2: Feature extraction
print("\n" + "=" * 60)
print("Example 2: Feature Extraction")
print("=" * 60)
print("""
from src.features.dino import get_feature_extractor
import torch

# Create feature extractor
extractor = get_feature_extractor(
    model_type="dino",
    model_name="vit_small_patch16_224.dino"
)

# Extract features from image
img = torch.randn(1, 3, 224, 224)  # Batch of images
features = extractor.extract_descriptors(
    img, 
    return_spatial=True
)
# features shape: [1, 384, 14, 14]
# 384 = feature dimension
# 14x14 = spatial grid
""")

# Example 3: Matching
print("\n" + "=" * 60)
print("Example 3: Matching")
print("=" * 60)
print("""
from src.matching.matcher import Matcher

# Create matcher
matcher = Matcher(
    matching_type="mutual_nn",  # Use mutual nearest neighbors
    use_ransac=True,            # Enable RANSAC filtering
    ransac_threshold=3.0
)

# Match descriptors
src_desc = src_features.squeeze(0)  # [D, H, W]
trg_desc = trg_features.squeeze(0)  # [D, H, W]
src_kps = torch.tensor([[50, 60], [100, 120]])  # Example keypoints

matched_src, matched_trg = matcher.match(
    src_desc, 
    trg_desc, 
    src_kps
)
# matched_src: matched source keypoints
# matched_trg: corresponding target keypoints
""")

# Example 4: Evaluation
print("\n" + "=" * 60)
print("Example 4: PCK Evaluation")
print("=" * 60)
print("""
from src.metrics.pck import PCKMetric

# Create PCK metric
pck_metric = PCKMetric(alphas=[0.05, 0.1, 0.15])

# Evaluate predictions
pred_kps = matched_trg  # Predicted keypoints
gt_kps = sample['trg_kps']  # Ground truth keypoints
img_size = (224, 224)

pck_metric.update(pred_kps, gt_kps, img_size)

# Get results
results = pck_metric.compute()
print(results)
# {'pck@0.05': 45.6, 'pck@0.1': 62.3, 'pck@0.15': 71.2}
""")

# Example 5: Complete pipeline
print("\n" + "=" * 60)
print("Example 5: Complete Pipeline")
print("=" * 60)
print("""
import torch
from src.datasets.spair import SPairDataset
from src.features.dino import get_feature_extractor
from src.matching.matcher import Matcher
from src.metrics.pck import PCKMetric
from utils.seed import set_seed

# Set seed for reproducibility
set_seed(42)

# Initialize components
dataset = SPairDataset(root_dir="data/SPair-71k", img_size=224)
extractor = get_feature_extractor("dino", "vit_small_patch16_224.dino")
matcher = Matcher(matching_type="mutual_nn")
pck_metric = PCKMetric(alphas=[0.1])

# Process one sample
sample = dataset[0]
src_img = sample['src_img'].unsqueeze(0)  # Add batch dim
trg_img = sample['trg_img'].unsqueeze(0)

# Extract features
with torch.no_grad():
    src_desc = extractor.extract_descriptors(src_img, return_spatial=True)
    trg_desc = extractor.extract_descriptors(trg_img, return_spatial=True)

# Match
matched_src, matched_trg = matcher.match(
    src_desc.squeeze(0),
    trg_desc.squeeze(0),
    sample['src_kps']
)

# Evaluate
pck_metric.update(matched_trg, sample['trg_kps'], (224, 224))
results = pck_metric.compute()
print(f"PCK@0.1: {results['pck@0.1']:.2f}%")
""")

# Example 6: Custom configuration
print("\n" + "=" * 60)
print("Example 6: Custom Configuration")
print("=" * 60)
print("""
# Create a custom config file (my_config.yaml):
dataset:
  root_dir: "data/SPair-71k"
  split: "test"
  category: "cat"
  img_size: 224

features:
  model_type: "dinov2"
  model_name: "base"  # Use larger model

matching:
  type: "mutual_nn"
  use_ransac: true
  ransac_threshold: 5.0

evaluation:
  alphas: [0.1]

logging:
  use_wandb: true
  project: "my-project"
  run_name: "dinov2-base-ransac"

seed: 42
device: "cuda"

# Run with custom config:
# python eval.py --config my_config.yaml
""")

print("\n" + "=" * 60)
print("For more information, see:")
print("  - README.md for full documentation")
print("  - QUICKSTART.md for getting started")
print("  - configs/ for example configurations")
print("=" * 60)
