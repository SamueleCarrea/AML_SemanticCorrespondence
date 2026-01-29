"""Model components."""

from .backbones import DINOv2Extractor, DINOv3Extractor, SAMImageEncoder
from .finetuner import FinetunableBackbone
from .loss import CorrespondenceLoss
from .config import BackboneConfig, BACKBONE_REGISTRY, get_backbone_config
from .unified_backbone import UnifiedBackbone
from .matcher import CorrespondenceMatcher
from .evaluator import UnifiedEvaluator

__all__ = [
    'DINOv2Extractor',
    'DINOv3Extractor', 
    'SAMImageEncoder',
    'FinetunableBackbone',
    'CorrespondenceLoss',
    'BackboneConfig',
    'BACKBONE_REGISTRY',
    'get_backbone_config',
    'UnifiedBackbone',
    'CorrespondenceMatcher',
    'UnifiedEvaluator',
]