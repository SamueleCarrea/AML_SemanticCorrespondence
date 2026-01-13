"""Vision Transformer backbones for semantic correspondence.

Supports DINOv2, DINOv3, and SAM with automatic padding for variable image sizes.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add parent directory to path for importing utils
sys.path.append(str(Path(__file__).parent.parent))
#from utils.image_utils import pad_to_patch_size, unpad_features


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


class _BaseExtractor(nn.Module):
    """Base class for feature extractors with automatic padding support."""
    
    stride: int

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__()
        self.device = _default_device(device)

    def extract_feats(
        self, image: torch.Tensor, return_padding: bool = False
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, Tuple[int, int, int, int]]:
        """Extract features from image.
        
        Args:
            image: (3, H, W) or (B, 3, H, W) image tensor
            return_padding: If True, also return padding used
            
        Returns:
            - feat_map: (B, C, H_feat, W_feat) feature map
            - stride: Patch size used
            - padding: (pad_left, pad_right, pad_top, pad_bottom) if return_padding=True
        """
        raise NotImplementedError

    @staticmethod
    def _normalize(feats: torch.Tensor) -> torch.Tensor:
        """L2-normalize features along channel dimension."""
        return F.normalize(feats, dim=1, eps=1e-6)


class DINOv2Extractor(_BaseExtractor):
    """DINOv2 Vision Transformer feature extractor.
    
    Automatically handles images of any size by padding to multiples of patch size.

    Args:
        variant: Hub name, e.g. 'dinov2_vitb14' (patch_size=14) or 'dinov2_vitl14'
        checkpoint_path: Optional path to local checkpoint (avoids hub download)
        device: Device to run on ('cuda', 'cpu', or None for auto)
        allow_hub_download: Allow downloading from torch.hub if checkpoint not found
        
    Example:
        >>> extractor = DINOv2Extractor('dinov2_vitb14', device='cuda')
        >>> img = torch.randn(1, 3, 345, 512)  # Variable size
        >>> feats, stride = extractor.extract_feats(img)
        >>> feats.shape  # (1, 768, 24, 36) - automatically padded and unpadded
        >>> stride  # 14
    """

    def __init__(
        self,
        variant: str = "dinov2_vitb14",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_hub_download: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.variant = variant
        self.model = self._load_model(variant, checkpoint_path, allow_hub_download)
        self.model.to(self.device)
        self.model.eval()
        self.stride = self._infer_stride(variant)
        
        print(f"✅ DINOv2Extractor loaded: {variant}")
        print(f"   Patch size (stride): {self.stride}")

    @staticmethod
    def _infer_stride(variant: str) -> int:
        """Extract patch size from model variant name."""
        match = re.search(r"(\d+)$", variant)
        if match:
            return int(match.group(1))
        return 14  # Default for ViT-B/14

    def _load_model(
        self, variant: str, checkpoint_path: Optional[str], allow_hub_download: bool
    ) -> nn.Module:
        """Load DINOv2 model from hub or local checkpoint."""
        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.join("checkpoints", checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path}. "
                    "Place a DINOv2 checkpoint in the checkpoints/ directory "
                    "or allow hub download."
                )
        try:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                variant,
                pretrained=checkpoint_path is None,
                trust_repo=True,
            )
        except Exception as exc:
            if checkpoint_path is None or not allow_hub_download:
                raise RuntimeError(
                    "Unable to load DINOv2 weights. "
                    "Provide a local checkpoint via `checkpoint_path`."
                ) from exc
            raise
        if checkpoint_path is not None:
            model = _load_state_dict(model, checkpoint_path)
        return model

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens and reshape to spatial feature map."""
        with torch.no_grad():
            outputs = self.model.forward_features(image)
            tokens = outputs.get("x_norm_patchtokens") or outputs.get(
                "x_norm_patch_tokens"
            )
            if tokens is None:
                tokens = outputs.get("patch_tokens")
            if tokens is None:
                raise RuntimeError(
                    "DINOv2 model did not return patch tokens; check model definition."
                )
            b, n, c = tokens.shape
            h = int(math.sqrt(n))
            w = n // h
            feat_map = tokens.transpose(1, 2).reshape(b, c, h, w)
        return feat_map

    def extract_feats(
        self, image: torch.Tensor, return_padding: bool = False
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, Tuple[int, int, int, int]]:
        """Extract dense features with automatic padding for variable image sizes.
        
        Args:
            image: (3, H, W) or (B, 3, H, W) normalized image
            return_padding: Return padding info (useful for keypoint adjustment)
            
        Returns:
            - feat_map: (B, C, H_feat, W_feat) normalized feature map
            - stride: Patch size (14 for DINOv2-B/14)
            - padding: (pad_left, pad_right, pad_top, pad_bottom) if return_padding=True
        """
        image = image.to(self.device)
        
        # Handle single image vs batch
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, 3, H, W)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = image.shape
        
        # Pad to multiple of patch size
        # Process each image in batch independently for padding
        padded_images = []
        paddings = []
        
        for i in range(B):
            img = image[i]  # (3, H, W)
            img_padded, padding = pad_to_patch_size(
                img, patch_size=self.stride, mode='constant', value=0.0
            )
            padded_images.append(img_padded)
            paddings.append(padding)
        
        # Stack back to batch
        image_padded = torch.stack(padded_images, dim=0)  # (B, 3, H', W')
        
        # Extract features from padded images
        feat_map_padded = self._forward_features(image_padded)  # (B, C, H'//stride, W'//stride)
        
        # Unpad features (assuming same padding for all images in batch)
        # For simplicity, we'll process each feature map individually
        feat_maps_unpadded = []
        for i in range(B):
            feat = feat_map_padded[i]  # (C, H', W')
            feat = feat.permute(1, 2, 0)  # (H', W', C)
            feat_unpadded = unpad_features(feat, paddings[i], self.stride)
            feat_unpadded = feat_unpadded.permute(2, 0, 1)  # (C, H, W)
            feat_maps_unpadded.append(feat_unpadded)
        
        feat_map = torch.stack(feat_maps_unpadded, dim=0)  # (B, C, H, W)
        
        # Normalize
        feat_map = self._normalize(feat_map)
        
        if squeeze_output:
            feat_map = feat_map.squeeze(0)  # (C, H, W)
        
        if return_padding:
            return feat_map, self.stride, paddings[0]  # Return first padding
        
        return feat_map, self.stride


class DINOv3Extractor(_BaseExtractor):
    """DINOv3 Vision Transformer feature extractor.
    
    Automatically handles images of any size by padding to multiples of patch size.

    Args:
        variant: Torch hub entry name, defaults to 'dinov3_vitb14'
        checkpoint_path: Optional local checkpoint path
        device: Device to run on
        allow_hub_download: Allow hub downloads
    """

    def __init__(
        self,
        variant: str = "dinov3_vitb14",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_hub_download: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.variant = variant
        self.model = self._load_model(variant, checkpoint_path, allow_hub_download)
        self.model.to(self.device)
        self.model.eval()
        self.stride = self._infer_stride(variant)
        
        print(f"✅ DINOv3Extractor loaded: {variant}")
        print(f"   Patch size (stride): {self.stride}")

    @staticmethod
    def _infer_stride(variant: str) -> int:
        match = re.search(r"(\d+)$", variant)
        if match:
            return int(match.group(1))
        return 14

    def _load_model(
        self, variant: str, checkpoint_path: Optional[str], allow_hub_download: bool
    ) -> nn.Module:
        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.join("checkpoints", checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path}. "
                    "Place a DINOv3 checkpoint in checkpoints/ or allow hub download."
                )
        try:
            model = torch.hub.load(
                "facebookresearch/dinov3",
                variant,
                pretrained=checkpoint_path is None,
                trust_repo=True,
            )
        except Exception as exc:
            if checkpoint_path is None or not allow_hub_download:
                raise RuntimeError(
                    "Unable to load DINOv3 weights. "
                    "Provide a local checkpoint via `checkpoint_path`."
                ) from exc
            raise
        if checkpoint_path is not None:
            model = _load_state_dict(model, checkpoint_path)
        return model

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs: Dict[str, torch.Tensor] = self.model.forward_features(image)
            tokens = outputs.get("x_norm_patchtokens") or outputs.get(
                "x_norm_patch_tokens"
            )
            if tokens is None:
                tokens = outputs.get("patch_tokens")
            if tokens is None:
                raise RuntimeError(
                    "DINOv3 model did not return patch tokens; check model definition."
                )
            b, n, c = tokens.shape
            h = int(math.sqrt(n))
            w = n // h
            feat_map = tokens.transpose(1, 2).reshape(b, c, h, w)
        return feat_map

    def extract_feats(
        self, image: torch.Tensor, return_padding: bool = False
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, Tuple[int, int, int, int]]:
        """Extract features with automatic padding (same logic as DINOv2)."""
        image = image.to(self.device)
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = image.shape[0]
        
        # Pad images
        padded_images = []
        paddings = []
        for i in range(B):
            img_padded, padding = pad_to_patch_size(
                image[i], patch_size=self.stride, mode='constant', value=0.0
            )
            padded_images.append(img_padded)
            paddings.append(padding)
        
        image_padded = torch.stack(padded_images, dim=0)
        
        # Extract and unpad features
        feat_map_padded = self._forward_features(image_padded)
        
        feat_maps_unpadded = []
        for i in range(B):
            feat = feat_map_padded[i].permute(1, 2, 0)
            feat_unpadded = unpad_features(feat, paddings[i], self.stride)
            feat_maps_unpadded.append(feat_unpadded.permute(2, 0, 1))
        
        feat_map = torch.stack(feat_maps_unpadded, dim=0)
        feat_map = self._normalize(feat_map)
        
        if squeeze_output:
            feat_map = feat_map.squeeze(0)
        
        if return_padding:
            return feat_map, self.stride, paddings[0]
        
        return feat_map, self.stride


class SAMImageEncoder(_BaseExtractor):
    """Segment Anything Model (SAM) image encoder feature extractor.
    
    Uses SAM's ViT encoder with automatic padding for variable image sizes.

    Args:
        variant: 'vit_b', 'vit_l' or 'vit_h' (SAM backbones)
        checkpoint_path: Optional local checkpoint
        device: Device to run on
        allow_hub_download: Allow hub downloads
        
    Note:
        SAM typically uses patch_size=16 for all variants.
    """

    def __init__(
        self,
        variant: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        allow_hub_download: bool = True,
    ) -> None:
        super().__init__(device=device)
        self.variant = variant
        self.model = self._load_model(variant, checkpoint_path, allow_hub_download)
        self.model.to(self.device)
        self.model.eval()
        self.stride = self._infer_stride()
        
        print(f"✅ SAMImageEncoder loaded: {variant}")
        print(f"   Patch size (stride): {self.stride}")

    def _load_model(
        self, variant: str, checkpoint_path: Optional[str], allow_hub_download: bool
    ) -> nn.Module:
        hub_name = f"sam_{variant}"
        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.join("checkpoints", checkpoint_path)
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found at {checkpoint_path}. "
                    "Place a SAM checkpoint in checkpoints/ or allow hub download."
                )
        try:
            model = torch.hub.load(
                "facebookresearch/segment-anything",
                hub_name,
                pretrained=checkpoint_path is None,
                trust_repo=True,
            )
        except Exception as exc:
            if checkpoint_path is None or not allow_hub_download:
                raise RuntimeError(
                    "Unable to load SAM image encoder. "
                    "Provide a local checkpoint via `checkpoint_path`."
                ) from exc
            raise
        if checkpoint_path is not None:
            model = _load_state_dict(model, checkpoint_path)
        return model

    def _infer_stride(self) -> int:
        """Infer patch size from SAM's patch_embed layer."""
        patch_embed = getattr(self.model.image_encoder, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if patch_size is None:
            return 16  # SAM default
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features using SAM image encoder."""
        with torch.no_grad():
            feat_map = self.model.image_encoder(image)
        return feat_map

    def extract_feats(
        self, image: torch.Tensor, return_padding: bool = False
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, Tuple[int, int, int, int]]:
        """Extract features with automatic padding (same logic as DINOv2/v3)."""
        image = image.to(self.device)
        
        if image.ndim == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B = image.shape[0]
        
        # Pad images
        padded_images = []
        paddings = []
        for i in range(B):
            img_padded, padding = pad_to_patch_size(
                image[i], patch_size=self.stride, mode='constant', value=0.0
            )
            padded_images.append(img_padded)
            paddings.append(padding)
        
        image_padded = torch.stack(padded_images, dim=0)
        
        # Extract features (SAM returns features directly in spatial format)
        feat_map_padded = self._forward_features(image_padded)  # (B, C, H', W')
        
        # Unpad features
        feat_maps_unpadded = []
        for i in range(B):
            feat = feat_map_padded[i].permute(1, 2, 0)  # (H', W', C)
            feat_unpadded = unpad_features(feat, paddings[i], self.stride)
            feat_maps_unpadded.append(feat_unpadded.permute(2, 0, 1))  # (C, H, W)
        
        feat_map = torch.stack(feat_maps_unpadded, dim=0)
        feat_map = self._normalize(feat_map)
        
        if squeeze_output:
            feat_map = feat_map.squeeze(0)
        
        if return_padding:
            return feat_map, self.stride, paddings[0]
        
        return feat_map, self.stride
