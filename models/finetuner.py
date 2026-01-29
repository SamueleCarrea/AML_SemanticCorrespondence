from __future__ import annotations
import torch
import torch.nn as nn
from .backbones import DINOv2Extractor, DINOv3Extractor, SAMImageEncoder

class FinetunableBackbone(nn.Module):
    """
    Unified wrapper:
    - Loads extractor (DINOv2 / DINOv3 / SAM)
    - Freezes all params
    - Unfreezes last N transformer blocks
    - Optionally enables gradient checkpointing (best-effort, depending on backbone impl)
    - Returns features as (B, H, W, D)
    """
    def __init__(
        self,
        backbone_name: str,
        num_layers_to_finetune: int,
        device: str,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_layers_to_finetune = num_layers_to_finetune
        self.device = device

        # -------------------------
        # Load extractor
        # -------------------------
        if backbone_name.startswith("dinov2"):
            self.extractor = DINOv2Extractor(variant=backbone_name, device=device)
            self._model_for_unfreeze = self.extractor.model
        elif backbone_name.startswith("dinov3"):
            self.extractor = DINOv3Extractor(variant=backbone_name, device=device)
            self._model_for_unfreeze = self.extractor.model
        elif backbone_name.startswith("sam"):
            variant = backbone_name.replace("sam_", "")  # vit_b / vit_l / vit_h
            self.extractor = SAMImageEncoder(variant=variant, device=device, allow_hub_download=True)
            self._model_for_unfreeze = self.extractor.model.image_encoder
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.stride = self.extractor.stride

        # Freeze all
        for p in self.extractor.parameters():
            p.requires_grad = False

        # Unfreeze last blocks
        self._unfreeze_last_blocks(num_layers_to_finetune)
        if num_layers_to_finetune > 0:
            self.extractor.train()
            
        # Enable gradient checkpointing (best-effort)
        self._enable_gradient_checkpointing(use_gradient_checkpointing)

        # Infer feature dim with a tiny forward
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 224, 224), device=self.device)
            feat_map, _ = self.extractor.extract_feats(dummy)
            self.feat_dim = int(feat_map.shape[-1])

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())

        print("\n Model summary:")
        print(f"  Backbone: {backbone_name}")
        print(f"  Stride:   {self.stride}")
        print(f"  Feat dim: {self.feat_dim}")
        print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({(n_trainable/n_total*100 if n_total else 0):.2f}%)\n")

    def _get_blocks(self):
        m = self._model_for_unfreeze
        if hasattr(m, "blocks"):
            return m.blocks
        if hasattr(m, "encoder") and hasattr(m.encoder, "layers"):
            return m.encoder.layers
        return None

    def _unfreeze_last_blocks(self, num_layers: int):
        if num_layers <= 0:
            print("\n Unfreezing: none (frozen backbone)\n")
            return

        blocks = self._get_blocks()
        if blocks is None:
            raise AttributeError("Cannot find transformer blocks to unfreeze in the selected backbone.")

        total = len(blocks)
        start = max(0, total - num_layers)
        print(f"\n Unfreezing last {num_layers} blocks: [{start}..{total-1}] out of {total}\n")

        for i in range(start, total):
            for p in blocks[i].parameters():
                p.requires_grad = True

    def _enable_gradient_checkpointing(self, enabled: bool):
        if not enabled:
            return

        m = self._model_for_unfreeze

        tried = []
        if hasattr(m, "gradient_checkpointing_enable"):
            tried.append("gradient_checkpointing_enable")
            try:
                m.gradient_checkpointing_enable()
                print("Gradient checkpointing enabled via .gradient_checkpointing_enable().")
                return
            except Exception as e:
                print(f"! gradient_checkpointing_enable failed: {e}")

        if hasattr(m, "set_grad_checkpointing"):
            tried.append("set_grad_checkpointing")
            try:
                m.set_grad_checkpointing(True)
                print("Gradient checkpointing enabled via .set_grad_checkpointing(True).")
                return
            except Exception as e:
                print(f"WARNING: set_grad_checkpointing failed: {e}")

        if hasattr(m, "set_gradient_checkpointing"):
            tried.append("set_gradient_checkpointing")
            try:
                m.set_gradient_checkpointing(True)
                print("Gradient checkpointing enabled via .set_gradient_checkpointing(True).")
                return
            except Exception as e:
                print(f"WARNING: set_gradient_checkpointing failed: {e}")

        if hasattr(m, "use_checkpoint"):
            tried.append("use_checkpoint")
            try:
                m.use_checkpoint = True
                print("Gradient checkpointing enabled via .use_checkpoint=True.")
                return
            except Exception as e:
                print(f"WARNING: use_checkpoint flag failed: {e}")

        print("WARNING: Gradient checkpointing requested but no supported API was found on this backbone.")
        if tried:
            print("  Tried:", tried)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat_map, _ = self.extractor.extract_feats(image)
        return feat_map