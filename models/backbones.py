import math
import os
import re
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _default_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_state_dict(model: nn.Module, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model


class _BaseExtractor(nn.Module):
    stride: int

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__()
        self.device = _default_device(device)

    def extract_feats(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        raise NotImplementedError

    @staticmethod
    def _normalize(feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(feats, dim=1, eps=1e-6)


class DINOv2Extractor(_BaseExtractor):
    """
    Wrapper around DINOv2 ViT models.

    Args:
        variant: hub name, e.g. ``dinov2_vitb14`` or ``dinov2_vitl14``.
        checkpoint_path: optional path inside ``checkpoints/`` to avoid hub download.
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

    @staticmethod
    def _infer_stride(variant: str) -> int:
        match = re.search(r"(\d+)$", variant)
        if match:
            return int(match.group(1))
        # Default to ViT-B/14 stride if parsing fails
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
        except Exception as exc:  # noqa: BLE001
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

    def extract_feats(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        image = image.to(self.device)
        feat_map = self._forward_features(image)
        feat_map = self._normalize(feat_map)
        return feat_map, self.stride


class DINOv3Extractor(_BaseExtractor):
    """
    Wrapper around DINOv3 models.

    Args:
        variant: torch hub entry name, defaults to ``dinov3_vitb14``.
        checkpoint_path: optional local checkpoint path.
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
        except Exception as exc:  # noqa: BLE001
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

    def extract_feats(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        image = image.to(self.device)
        feat_map = self._forward_features(image)
        feat_map = self._normalize(feat_map)
        return feat_map, self.stride


class SAMImageEncoder(_BaseExtractor):
    """
    Wrapper around the Segment Anything Model (SAM) image encoder.

    Args:
        variant: ``vit_b``, ``vit_l`` or ``vit_h`` corresponding to SAM backbones.
        checkpoint_path: optional local checkpoint filename or path.
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
        except Exception as exc:  # noqa: BLE001
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
        patch_embed = getattr(self.model.image_encoder, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", None)
        if patch_size is None:
            return 16
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    def _forward_features(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat_map = self.model.image_encoder(image)
        return feat_map

    def extract_feats(self, image: torch.Tensor) -> Tuple[torch.Tensor, int]:
        image = image.to(self.device)
        feat_map = self._forward_features(image)
        feat_map = self._normalize(feat_map)
        return feat_map, self.stride
