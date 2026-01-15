"""Fine-tunable backbone wrappers for semantic correspondence."""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
from .backbones import DINOv2Extractor, DINOv3Extractor


class FinetunableBackbone(nn.Module):
    """Wrapper to make backbone partially trainable.
    
    Args:
        backbone_name: Name of the backbone ('dinov2_vitb14', 'dinov3_vitb16')
        num_layers_to_finetune: Number of last transformer blocks to unfreeze
        device: Device to run on
    """

    def __init__(
        self,
        backbone_name: str,
        num_layers_to_finetune: int = 2,
        device: Optional[str] = None
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_layers_to_finetune = num_layers_to_finetune
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load frozen backbone
        if 'dinov2' in backbone_name:
            self.extractor = DINOv2Extractor(
                variant=backbone_name,
                device=self.device,
                allow_hub_download=True
            )
        elif 'dinov3' in backbone_name:
            self.extractor = DINOv3Extractor(
                variant=backbone_name,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.stride = self.extractor.stride

        # Freeze all parameters first
        for param in self.extractor.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer blocks
        self._unfreeze_last_layers(num_layers_to_finetune)

        # Count trainable parameters
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())

        print(f"\n Trainable parameters:")
        print(f"   Total: {n_total:,}")
        print(f"   Trainable: {n_trainable:,} ({n_trainable/n_total*100:.2f}%)")

    def _unfreeze_last_layers(self, num_layers: int):
        """Unfreeze last N transformer blocks."""
        model = self.extractor.model

        # Access transformer blocks
        if hasattr(model, 'blocks'):
            blocks = model.blocks
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            blocks = model.encoder.layers
        else:
            raise AttributeError("Cannot find transformer blocks in model")

        total_blocks = len(blocks)
        start_idx = max(0, total_blocks - num_layers)

        print(f"\n Unfreezing blocks {start_idx} to {total_blocks-1} (total: {total_blocks})")

        for i in range(start_idx, total_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract features (with gradient if training).
        
        Args:
            image: (B, 3, H, W)
            
        Returns:
            features: (B, H, W, D)
        """
        feat_map, stride = self.extractor.extract_feats(image)
        features = feat_map.permute(0, 2, 3, 1)  # (B, H, W, D)
        return features

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.extract_features(image)