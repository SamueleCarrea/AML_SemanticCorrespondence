"""Image processing utilities for semantic correspondence.

Handles variable image sizes and multiple backbone architectures with different patch sizes.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
#from models.backbones import DINOv2Extractor, DINOv3Extractor, SAMImageEncoder


def make_divisible(x: int, divisor: int) -> int:
    """Round up x to nearest multiple of divisor."""
    return ((x + divisor - 1) // divisor) * divisor


def pad_to_patch_size(
    image: torch.Tensor,
    patch_size: int = 14,
    mode: str = "constant",
    value: float = 0.0
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad image to make dimensions divisible by patch_size."""
    C, H, W = image.shape
    
    H_new = make_divisible(H, patch_size)
    W_new = make_divisible(W, patch_size)
    
    pad_h = H_new - H
    pad_w = W_new - W
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    
    if mode == "constant":
        padded = F.pad(image, padding, mode=mode, value=value)
    else:
        padded = F.pad(image, padding, mode=mode)
    
    return padded, padding


def unpad_features(
    features: torch.Tensor,
    padding: Tuple[int, int, int, int],
    patch_size: int = 14
) -> torch.Tensor:
    """Remove padding from feature maps."""
    pad_left, pad_right, pad_top, pad_bottom = padding
    
    patch_pad_left = pad_left // patch_size
    patch_pad_right = pad_right // patch_size
    patch_pad_top = pad_top // patch_size
    patch_pad_bottom = pad_bottom // patch_size
    
    H, W, D = features.shape
    
    h_start = patch_pad_top
    h_end = H - patch_pad_bottom if patch_pad_bottom > 0 else H
    w_start = patch_pad_left
    w_end = W - patch_pad_right if patch_pad_right > 0 else W
    
    return features[h_start:h_end, w_start:w_end, :]


# Test con diverse backbone

img = torch.randn(1, 3, 345, 512)  # Dimensione non standard

# DINOv2 (patch_size=14)
dinov2 = DINOv2Extractor('dinov2_vitb14', device='cuda')
feats, stride = dinov2.extract_feats(img)

# SAM (patch_size=16)
sam = SAMImageEncoder('vit_b', device='cuda')
feats, stride = sam.extract_feats(img)