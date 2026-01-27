"""Loss functions for semantic correspondence fine-tuning."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrespondenceLoss(nn.Module):
    """Loss function for semantic correspondence fine-tuning.
    
    Supports multiple loss types:
    - 'cosine': Maximize cosine similarity between corresponding features
    - 'l2': Minimize L2 distance between corresponding features
    - 'contrastive': InfoNCE-style contrastive loss with negative sampling
    
    Args:
        loss_type: Type of loss ('cosine', 'l2', 'contrastive')
        temperature: Temperature for contrastive loss
    """

    def __init__(
        self,
        loss_type: str = 'cosine',
        temperature: float = 0.1
    ):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature

    def forward(
        self,
        src_features: torch.Tensor,
        tgt_features: torch.Tensor,
        src_kps: torch.Tensor,
        tgt_kps: torch.Tensor,
        valid_mask: torch.Tensor,
        patch_size: int
    ) -> torch.Tensor:
        """Compute correspondence loss.

        Args:
            src_features: (B, H_s, W_s, D)
            tgt_features: (B, H_t, W_t, D)
            src_kps: (B, N, 2) source keypoints in pixel coords
            tgt_kps: (B, N, 2) target keypoints in pixel coords
            valid_mask: (B, N) valid keypoint mask
            patch_size: Patch size of the backbone

        Returns:
            loss: Scalar tensor
        """
        B = src_features.shape[0]
        total_loss = 0.0
        n_valid = 0

        for b in range(B):
            src_feat = src_features[b]  # (H_s, W_s, D)
            tgt_feat = tgt_features[b]  # (H_t, W_t, D)
            src_kp = src_kps[b]  # (N, 2)
            tgt_kp = tgt_kps[b]  # (N, 2)
            valid = valid_mask[b]  # (N,)

            if valid.sum() == 0:
                continue

            # Filter valid keypoints
            src_kp_valid = src_kp[valid]
            tgt_kp_valid = tgt_kp[valid]

            src_kp_patch = (src_kp_valid / patch_size).long()
            tgt_kp_patch = (tgt_kp_valid / patch_size).long()

            H_s, W_s, D = src_feat.shape
            H_t, W_t, _ = tgt_feat.shape

            # Clamp coordinates
            src_kp_patch[:, 0] = src_kp_patch[:, 0].clamp(0, W_s - 1)
            src_kp_patch[:, 1] = src_kp_patch[:, 1].clamp(0, H_s - 1)
            tgt_kp_patch[:, 0] = tgt_kp_patch[:, 0].clamp(0, W_t - 1)
            tgt_kp_patch[:, 1] = tgt_kp_patch[:, 1].clamp(0, H_t - 1)

            # Extract features at keypoint locations
            N = src_kp_valid.shape[0]
            for i in range(N):
                src_x, src_y = src_kp_patch[i]
                tgt_x, tgt_y = tgt_kp_patch[i]

                src_vec = src_feat[src_y, src_x]  # (D,)
                tgt_vec = tgt_feat[tgt_y, tgt_x]  # (D,)

                if self.loss_type == 'cosine':
                    # Maximize cosine similarity
                    similarity = F.cosine_similarity(
                        src_vec.unsqueeze(0), tgt_vec.unsqueeze(0), dim=1
                    )
                    loss = 1.0 - similarity

                elif self.loss_type == 'l2':
                    # Minimize L2 distance
                    loss = F.mse_loss(src_vec, tgt_vec)

                elif self.loss_type == 'contrastive':
                    # Positive: corresponding point
                    pos_sim = F.cosine_similarity(
                        src_vec.unsqueeze(0), tgt_vec.unsqueeze(0), dim=1
                    )

                    # Negatives: sample random points from target
                    neg_indices = torch.randint(
                        0, H_t * W_t, (8,), device=src_vec.device
                    )
                    tgt_flat = tgt_feat.reshape(-1, D)
                    neg_vecs = tgt_flat[neg_indices]  # (8, D)

                    neg_sim = F.cosine_similarity(
                        src_vec.unsqueeze(0).expand(8, -1),
                        neg_vecs,
                        dim=1
                    )

                    # InfoNCE-style loss
                    pos_exp = torch.exp(pos_sim / self.temperature)
                    neg_exp = torch.exp(neg_sim / self.temperature).sum()

                    loss = -torch.log(pos_exp / (pos_exp + neg_exp))

                else:
                    raise ValueError(f"Unknown loss type: {self.loss_type}")

                total_loss += loss
                n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=src_features.device, requires_grad=True)

        return total_loss / n_valid