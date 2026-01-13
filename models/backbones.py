"""Vision Transformer backbones for semantic correspondence.

Supports DINOv2, DINOv3, and SAM with automatic padding for variable image sizes.
"""
import sys
import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple


# ============================================================================
# Padding Utilities
# ============================================================================

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
    if image.dim() == 3:
        C, H, W = image.shape
    elif image.dim() == 4:
        B, C, H, W = image.shape
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")
    
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
    """Remove padding from feature maps.
    
    Args:
        features: (B, C, H, W) feature tensor
        padding: (pad_left, pad_right, pad_top, pad_bottom)
        patch_size: Stride/patch size
        
    Returns:
        Unpadded features (B, C, H_orig, W_orig)
    """
    pad_left, pad_right, pad_top, pad_bottom = padding
    
    patch_pad_left = pad_left // patch_size
    patch_pad_right = pad_right // patch_size
    patch_pad_top = pad_top // patch_size
    patch_pad_bottom = pad_bottom // patch_size
    
    B, C, H, W = features.shape
    
    h_start = patch_pad_top
    h_end = H - patch_pad_bottom if patch_pad_bottom > 0 else H
    w_start = patch_pad_left
    w_end = W - patch_pad_right if patch_pad_right > 0 else W
    
    return features[:, :, h_start:h_end, w_start:w_end]


# ============================================================================
# Helper Functions
# ============================================================================

def _default_device(device: Optional[str] = None) -> torch.device:
    """Get default device (CUDA if available, else CPU)."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model


# ============================================================================
# Base Extractor Class
# ============================================================================

class _BaseExtractor(nn.Module):
    """Base class for feature extractors with automatic padding support."""
    
    stride: int

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__()
        self.device = _default_device(device)
        self._last_img_shape = None  # Store padded image shape

    def extract_feats(
        self, image: torch.Tensor, return_padding: bool = False
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, Tuple[int, int, int, int]]:
        """Extract features with automatic padding."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        padded_img, padding = pad_to_patch_size(image, patch_size=self.stride)
        
        self._last_img_shape = padded_img.shape
        
        features = self._forward_features(padded_img)
        features = unpad_features(features, padding, patch_size=self.stride)
        
        if return_padding:
            return features, self.stride, padding
        return features, self.stride

    @staticmethod
    def _infer_patch_size(model) -> int:
        """Infer patch size from model architecture."""
        if hasattr(model, 'patch_embed'):
            if hasattr(model.patch_embed, 'patch_size'):
                ps = model.patch_embed.patch_size
                return ps[0] if isinstance(ps, (tuple, list)) else ps
        return 14  # Default fallback


# ============================================================================
# DINOv2 Extractor
# ============================================================================

class DINOv2Extractor(_BaseExtractor):
    """DINOv2 Vision Transformer feature extractor."""

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_hub_download: bool = True,
    ) -> None:
        super().__init__(device)
        self.variant = variant
        self.model = self._load_model(variant, checkpoint_path, allow_hub_download)
        self.model.eval()
        self.model.to(self.device)
        self.stride = self._infer_patch_size(self.model)

    def _load_model(
        self, variant: str, checkpoint_path: Optional[str], allow_hub_download: bool
    ) -> nn.Module:
        """Load DINOv2 model."""
        return torch.hub.load('facebookresearch/dinov2', variant, pretrained=True)


    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINOv2."""
        output = self.model.forward_features(image)
        patch_tokens = output['x_norm_patchtokens']
        
        B, N, D = patch_tokens.shape
        
        if self._last_img_shape is not None:
            _, _, H_img, W_img = self._last_img_shape
            H = H_img // self.stride
            W = W_img // self.stride
            
            if H * W != N:
                H = W = int(math.sqrt(N))
                if H * W != N:
                    raise RuntimeError(f"Cannot reshape {N} patches")
        else:
            H = W = int(math.sqrt(N))
        
        features = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return features


# ============================================================================
# DINOv3 Extractor
# ============================================================================

class DINOv3Extractor(_BaseExtractor):
    """DINOv3 Vision Transformer feature extractor."""

    def __init__(
        self,
        variant: str = "dinov3_vitb16",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_hub_download: bool = True,
    ) -> None:
        super().__init__(device)
        self.variant = variant
        self.model = self._load_model(variant, checkpoint_path, allow_hub_download)
        self.model.eval()
        self.model.to(self.device)
        self.stride = self._infer_patch_size(self.model)

    def _load_model(
        self, variant: str
    ) -> nn.Module:
        """Load DINOv3 model"""
        model = torch.hub.load('facebookresearch/dinov3', variant, pretrained=False)
        return _load_state_dict(model, '/content/drive/MyDrive/AML/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
        
    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINOv3."""
        output = self.model.forward_features(image)
        patch_tokens = output['x_norm_patchtokens']
        
        B, N, D = patch_tokens.shape
        
        # Calculate H and W from padded image shape
        if self._last_img_shape is not None:
            _, _, H_img, W_img = self._last_img_shape
            H = H_img // self.stride
            W = W_img // self.stride
        else:
            H = W = int(math.sqrt(N))
        
        if H * W != N:
            raise RuntimeError(
                f"Dimension mismatch: H={H}, W={W}, H*W={H*W}, but N={N}"
            )
        
        features = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        return features


# ============================================================================
# SAM Image Encoder
# ============================================================================

class SAMImageEncoder(_BaseExtractor):
    """Segment Anything Model (SAM) image encoder."""

    def __init__(
        self,
        variant: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_hub_download: bool = True,
    ) -> None:
        super().__init__(device)
        self.variant = variant
        self.model = self._load_model(variant, checkpoint_path, allow_hub_download)
        self.model.eval()
        self.model.to(self.device)
        self.stride = 16  # SAM uses patch_size=16

    def _load_model(
        self, variant: str, checkpoint_path: Optional[str], allow_hub_download: bool
    ) -> nn.Module:
        """Load SAM model."""
        try:
            from segment_anything import sam_model_registry
            
            if checkpoint_path and Path(checkpoint_path).exists():
                return sam_model_registry[variant](checkpoint=checkpoint_path)
            elif allow_hub_download:
                urls = {
                    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
                    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
                    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
                }
                checkpoint_path = f'/tmp/sam_{variant}.pth'
                if not Path(checkpoint_path).exists():
                    torch.hub.download_url_to_file(urls[variant], checkpoint_path)
                return sam_model_registry[variant](checkpoint=checkpoint_path)
            else:
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        except ImportError:
            raise ImportError("SAM requires: pip install git+https://github.com/facebookresearch/segment-anything.git")

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through SAM encoder."""
        features = self.model.image_encoder(image)
        return features
