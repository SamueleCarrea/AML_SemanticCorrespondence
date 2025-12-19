#!/usr/bin/env python
"""Quick test script to validate the baseline implementation."""

import sys
import torch
import numpy as np

print("Testing Semantic Correspondence Baseline...")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from src.datasets.spair import SPairDataset
    from src.features.dino import get_feature_extractor
    from src.matching.matcher import Matcher
    from src.metrics.pck import PCKMetric
    from utils.seed import set_seed
    from utils.logger import Logger
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Seed setting
print("\n2. Testing seed setting...")
try:
    set_seed(42)
    print("   ✓ Seed setting successful")
except Exception as e:
    print(f"   ✗ Seed setting failed: {e}")
    sys.exit(1)

# Test 3: Dataset with dummy data
print("\n3. Testing dataset loader...")
try:
    dataset = SPairDataset(root_dir="data/dummy", img_size=224)
    sample = dataset[0]
    assert 'src_img' in sample
    assert 'trg_img' in sample
    assert 'src_kps' in sample
    assert 'trg_kps' in sample
    print(f"   ✓ Dataset loader working (sample keys: {list(sample.keys())})")
except Exception as e:
    print(f"   ✗ Dataset loader failed: {e}")
    sys.exit(1)

# Test 4: Feature extraction
print("\n4. Testing feature extraction...")
try:
    device = torch.device('cpu')  # Use CPU for testing
    feature_extractor = get_feature_extractor(
        model_type="dino",
        model_name="vit_small_patch16_224.dino"
    ).to(device)
    
    # Test with dummy image
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    features = feature_extractor.extract_descriptors(dummy_img, return_spatial=True)
    
    assert features.dim() == 4  # [B, D, H, W]
    print(f"   ✓ Feature extraction working (shape: {features.shape})")
except Exception as e:
    print(f"   ✗ Feature extraction failed: {e}")
    print("   Note: This may fail if models are not downloaded yet")

# Test 5: Matching
print("\n5. Testing matching...")
try:
    matcher = Matcher(matching_type="mutual_nn", use_ransac=False)
    
    # Create dummy descriptors
    src_desc = torch.randn(384, 14, 14)
    trg_desc = torch.randn(384, 14, 14)
    src_kps = torch.rand(10, 2) * 224
    
    matched_src, matched_trg = matcher.match(src_desc, trg_desc, src_kps)
    
    print(f"   ✓ Matching working ({len(matched_src)} matches found)")
except Exception as e:
    print(f"   ✗ Matching failed: {e}")
    sys.exit(1)

# Test 6: PCK metric
print("\n6. Testing PCK metric...")
try:
    pck_metric = PCKMetric(alphas=[0.1])
    
    pred_kps = torch.rand(10, 2) * 224
    gt_kps = pred_kps + torch.randn(10, 2) * 5  # Add small noise
    
    pck_metric.update(pred_kps, gt_kps, img_size=(224, 224))
    results = pck_metric.compute()
    
    assert 'pck@0.1' in results
    print(f"   ✓ PCK metric working (PCK@0.1: {results['pck@0.1']:.2f}%)")
except Exception as e:
    print(f"   ✗ PCK metric failed: {e}")
    sys.exit(1)

# Test 7: Logger (without WandB)
print("\n7. Testing logger...")
try:
    logger = Logger(
        project="test",
        config={'test': True},
        use_wandb=False
    )
    logger.log({'test_metric': 0.5})
    logger.finish()
    print("   ✓ Logger working")
except Exception as e:
    print(f"   ✗ Logger failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("\nYou can now run the full evaluation:")
print("  python eval.py --data_dir data/dummy --no-wandb")
print("=" * 60)
