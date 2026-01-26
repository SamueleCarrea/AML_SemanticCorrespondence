"""Unified backbone interface for feature extraction."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

from models.config import get_backbone_config
from models.backbones import DINOv2Extractor, DINOv3Extractor, SAMImageEncoder


class UnifiedBackbone(nn.Module):
    """Unified interface for all vision backbones."""

    def __init__(
        self,
        backbone_choice: str,
        finetune_choice: bool = False,
        checkpoint_dir: str = None,
        device: str = 'cuda'
    ):
        """Initialize backbone with optional finetuned weights.
        
        Args:
            backbone_choice: 'dinov2', 'dinov3', or 'sam'
            finetune_choice: If True, load finetuned weights
            checkpoint_dir: Directory containing finetuned checkpoints
            device: Device to load model on ('cuda' or 'cpu')
        """
        super().__init__()
        
        self.config = get_backbone_config(backbone_choice)
        self.backbone_choice = backbone_choice
        self.finetune_choice = finetune_choice
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        # Load extractor
        self._load_extractor()
        
    def _load_extractor(self):
        """Instantiate and load the appropriate extractor."""
        
        print(f"→ Loading {self.config.name}...", end='')
        
        # Create extractor based on type
        if self.config.type == 'dinov2':
            self.extractor = DINOv2Extractor(
                variant=self.config.hub_name,
                device=self.device
            )
        elif self.config.type == 'dinov3':
            self.extractor = DINOv3Extractor(
                variant=self.config.hub_name,
                device=self.device
            )
        elif self.config.type == 'sam':
            self.extractor = SAMImageEncoder(
                variant=self.config.hub_name,
                device=self.device,
                allow_hub_download=True
            )
        else:
            raise NotImplementedError(f"Type {self.config.type} not implemented")
        
        # Load finetuned weights if requested
        if self.finetune_choice and self.checkpoint_dir:
            self._load_finetuned_weights()
        
        print(f" ✓")
        print(f"   • Patch size: {self.extractor.stride}")
        print(f"   • Embedding: {self.config.embed_dim}D")
        if self.finetune_choice:
            print(f"   • Weights: finetuned")
        
    def _load_finetuned_weights(self):
        """Load finetuned checkpoint if available."""
        
        checkpoint_name = f"finetuned_{self.backbone_choice}_*.pt"
        checkpoint_dir_path = Path(self.checkpoint_dir)
        matching_files = list(checkpoint_dir_path.glob(checkpoint_name))

        if not matching_files:
            print(f"Finetuned checkpoint not found: {checkpoint_pattern}")
            print(f"Using pretrained weights instead")
            return

        print("/nFiles founded: \n")
        for f in matching_files:
            print(f"{f}\n")
            checkpoint_path = matching_files[0]
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                print(f"   (loaded from {checkpoint_path.name})")
                if "best_pck@0.10" in checkpoint:
                    print(f"    (PCK@0.10: {checkpoint['best_pck@0.10']:.2f})")
            else:
                state_dict = checkpoint
                print(f"   (loaded from {checkpoint_path.name})")
            self.extractor.load_state_dict(state_dict, strict=False)
                    
        except Exception as e:
            print(f"   ⚠ Failed to load finetuned weights: {e}")
            print(f"   Using pretrained weights instead")

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract dense features from image.
        
        Args:
            image: (B, 3, H, W) tensor
            
        Returns:
            features: (B, H_patches, W_patches, D) tensor
        """
        feat_map, stride = self.extractor.extract_feats(image)
        return feat_map
