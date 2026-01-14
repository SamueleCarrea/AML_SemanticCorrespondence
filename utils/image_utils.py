"""Image processing utilities for semantic correspondence.

Note: Padding functions moved to models/backbones.py to avoid circular imports.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# Le funzioni pad_to_patch_size e unpad_features sono ora in models/backbones.py
# Puoi importarle così se serve:
# from models.backbones import pad_to_patch_size, unpad_features

def make_divisible(x: int, divisor: int) -> int:
    """Round up x to nearest multiple of divisor."""
    return ((x + divisor - 1) // divisor) * divisor

# Test con diverse backbone

img = torch.randn(1, 3, 345, 512)  # Dimensione non standard

# DINOv2 (patch_size=14)
# dinov2 = DINOv2Extractor('dinov2_vitb14', device='cuda')
# feats, stride = dinov2.extract_feats(img)

# SAM (patch_size=16)
# sam = SAMImageEncoder('vit_b', device='cuda')
# feats, stride = sam.extract_feats(img)