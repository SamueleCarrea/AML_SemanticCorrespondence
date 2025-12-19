"""Feature extraction using DINO and DINOv2 models."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import timm


class DinoFeatureExtractor(nn.Module):
    """DINO/DINOv2 feature extractor."""
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        layer: int = -1,
        facet: str = "token",
        use_cls: bool = False,
    ):
        """Initialize DINO feature extractor.
        
        Args:
            model_name: Name of the model (DINO or DINOv2)
            layer: Which layer to extract features from (-1 for last layer)
            facet: Feature facet to extract ('token', 'key', 'query', 'value')
            use_cls: Whether to include CLS token in features
        """
        super().__init__()
        
        self.model_name = model_name
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        
        # Load pretrained model using timm
        # Supported models: vit_small_patch16_224.dino, vit_base_patch16_224.dino, etc.
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        
        # Get patch size and feature dimension
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.feat_dim = self.model.embed_dim
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Extract features from images.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            features: Extracted features [B, N, D] where N is number of patches
            h: Height of feature map
            w: Width of feature map
        """
        B, C, H, W = x.shape
        
        # Calculate feature map size
        h = H // self.patch_size
        w = W // self.patch_size
        
        with torch.no_grad():
            # Extract features using forward_features
            features = self.model.forward_features(x)
            
            if not self.use_cls:
                # Remove CLS token (first token)
                features = features[:, 1:, :]
            
        return features, h, w
    
    def extract_descriptors(
        self,
        x: torch.Tensor,
        return_spatial: bool = True
    ) -> torch.Tensor:
        """Extract dense descriptors from images.
        
        Args:
            x: Input images [B, 3, H, W]
            return_spatial: Whether to return spatial feature map
            
        Returns:
            descriptors: Features [B, D, H, W] if return_spatial else [B, N, D]
        """
        features, h, w = self.forward(x)
        
        if return_spatial:
            # Reshape to spatial feature map [B, D, H, W]
            B = features.shape[0]
            features = features.permute(0, 2, 1).reshape(B, self.feat_dim, h, w)
        
        return features


class Dinov2FeatureExtractor(nn.Module):
    """DINOv2 feature extractor using torch.hub."""
    
    def __init__(
        self,
        model_size: str = "small",
        layer: int = -1,
        facet: str = "token",
        use_cls: bool = False,
    ):
        """Initialize DINOv2 feature extractor.
        
        Args:
            model_size: Size of the model ('small', 'base', 'large', 'giant')
            layer: Which layer to extract features from (-1 for last layer)
            facet: Feature facet to extract ('token', 'key', 'query', 'value')
            use_cls: Whether to include CLS token in features
        """
        super().__init__()
        
        self.model_size = model_size
        self.layer = layer
        self.facet = facet
        self.use_cls = use_cls
        
        # Map model size to model name
        model_map = {
            'small': 'dinov2_vits14',
            'base': 'dinov2_vitb14',
            'large': 'dinov2_vitl14',
            'giant': 'dinov2_vitg14',
        }
        
        model_name = model_map.get(model_size, 'dinov2_vits14')
        
        try:
            # Try to load DINOv2 from torch.hub
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        except (RuntimeError, ConnectionError, OSError) as e:
            # Fallback to timm if torch.hub fails
            print(f"Failed to load {model_name} from torch.hub ({e}), using timm fallback")
            self.model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True)
        
        self.model.eval()
        
        # Get patch size and feature dimension
        self.patch_size = 14  # DINOv2 uses 14x14 patches
        self.feat_dim = self.model.embed_dim if hasattr(self.model, 'embed_dim') else 384
        
        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Extract features from images.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            features: Extracted features [B, N, D]
            h: Height of feature map
            w: Width of feature map
        """
        B, C, H, W = x.shape
        
        # Calculate feature map size
        h = H // self.patch_size
        w = W // self.patch_size
        
        with torch.no_grad():
            # Extract features
            if hasattr(self.model, 'forward_features'):
                features = self.model.forward_features(x)
            else:
                features = self.model(x)
            
            if not self.use_cls:
                # Remove CLS token (first token)
                features = features[:, 1:, :]
        
        return features, h, w
    
    def extract_descriptors(
        self,
        x: torch.Tensor,
        return_spatial: bool = True
    ) -> torch.Tensor:
        """Extract dense descriptors from images.
        
        Args:
            x: Input images [B, 3, H, W]
            return_spatial: Whether to return spatial feature map
            
        Returns:
            descriptors: Features [B, D, H, W] if return_spatial else [B, N, D]
        """
        features, h, w = self.forward(x)
        
        if return_spatial:
            # Reshape to spatial feature map [B, D, H, W]
            B = features.shape[0]
            features = features.permute(0, 2, 1).reshape(B, self.feat_dim, h, w)
        
        return features


def get_feature_extractor(
    model_type: str = "dino",
    model_name: str = "vit_small_patch16_224.dino",
    **kwargs
) -> nn.Module:
    """Factory function to get feature extractor.
    
    Args:
        model_type: Type of model ('dino' or 'dinov2')
        model_name: Name/size of the model
        **kwargs: Additional arguments for the extractor
        
    Returns:
        Feature extractor module
    """
    if model_type.lower() == "dino":
        return DinoFeatureExtractor(model_name=model_name, **kwargs)
    elif model_type.lower() == "dinov2":
        return Dinov2FeatureExtractor(model_size=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
