from dataclasses import dataclass
from typing import Dict


@dataclass
class BackboneConfig:
    """Configuration for a vision backbone."""
    name: str
    patch_size: int
    embed_dim: int
    hub_name: str
    type: str


# Global registry mapping short names to configurations
BACKBONE_REGISTRY: Dict[str, BackboneConfig] = {
    'dinov2': BackboneConfig(
        name='DINOv2-ViT-B/14',
        patch_size=14,
        embed_dim=768,
        hub_name='dinov2_vitb14',
        type='dinov2'
    ),
    'dinov3': BackboneConfig(
        name='DINOv3-ViT-B/16',
        patch_size=16,
        embed_dim=768,
        hub_name='dinov3_vitb16',
        type='dinov3'
    ),
    'sam': BackboneConfig(
        name='SAM-ViT-B',
        patch_size=16,
        embed_dim=768,
        hub_name='vit_b',
        type='sam'
    ),
}


def get_backbone_config(backbone_choice: str) -> BackboneConfig:
    """Get backbone configuration from registry.
    
    Args:
        backbone_choice: Short name ('dinov2', 'dinov3', 'sam')
        
    Returns:
        BackboneConfig instance
        
    Raises:
        ValueError: If backbone_choice not in registry
    """
    if backbone_choice not in BACKBONE_REGISTRY:
        available = ', '.join(BACKBONE_REGISTRY.keys())
        raise ValueError(
            f"Unknown backbone: '{backbone_choice}'. "
            f"Available options: {available}"
        )
    
    return BACKBONE_REGISTRY[backbone_choice]
