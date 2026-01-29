"""Semantic correspondence matcher."""

import torch
import torch.nn.functional as F


class CorrespondenceMatcher:
    """Training-free baseline matcher using cosine similarity."""

    def __init__(self, backbone, use_soft_argmax: bool = False):
        """Initialize matcher.
        
        Args:
            backbone: UnifiedBackbone instance
            use_soft_argmax: If True, use soft argmax (task 3)
                            If False, use hard argmax (task 1).
        """
        self.backbone = backbone
        self.device = backbone.device
        self.use_soft_argmax = use_soft_argmax
    @torch.no_grad()
    def match(
        self,
        src_img: torch.Tensor,
        tgt_img: torch.Tensor,
        src_kps: torch.Tensor
    ) -> torch.Tensor:
        """Find correspondences for source keypoints.
        
        Args:
            src_img: (1, 3, H, W) source image
            tgt_img: (1, 3, H, W) target image
            src_kps: (N, 2) source keypoints in pixel coords (x, y)
            
        Returns:
            tgt_kps_pred: (N, 2) predicted target keypoints
        """
        # Extract features (backbone handles padding/unpadding)
        src_feat = self.backbone.extract_features(src_img)[0]  # (H_s, W_s, D)
        tgt_feat = self.backbone.extract_features(tgt_img)[0]  # (H_t, W_t, D)
        H_s, W_s, D = src_feat.shape
        _, W_t, _ = tgt_feat.shape
        patch_size = self.backbone.config.patch_size


        # Convert keypoint coords to patch indices
        src_kps = src_kps.to(self.device)
        src_kps_patch = (src_kps / patch_size).long()
        src_kps_patch[:, 0] = src_kps_patch[:, 0].clamp(0, W_s - 1)
        src_kps_patch[:, 1] = src_kps_patch[:, 1].clamp(0, H_s - 1)

        # Match each keypoint
        N = src_kps.shape[0]
        tgt_kps_pred = torch.zeros(N, 2, device=src_kps.device)

        for i in range(N):
            x = int(src_kps_patch[i, 0].item())
            y = int(src_kps_patch[i, 1].item())
            src_vec = src_feat[y, x]
            # Cosine similarity
            similarity = F.cosine_similarity(
                src_vec.view(1, 1, 1, D),
                tgt_feat.unsqueeze(0),
                dim=-1
            ).squeeze(0)  # (H_t, W_t)

            # Matching strategy
            if self.use_soft_argmax:
                pred_y, pred_x = self._soft_argmax(similarity)
            else:
                # Hard argmax (task 1)
                max_idx = similarity.flatten().argmax()
                pred_y = max_idx // W_t
                pred_x = max_idx % W_t

            # Convert back to pixel coordinates
            tgt_kps_pred[i, 0] = pred_x * patch_size + patch_size / 2.0
            tgt_kps_pred[i, 1] = pred_y * patch_size + patch_size / 2.0

        return tgt_kps_pred
    
    def _soft_argmax(self, similarity: torch.Tensor, window_size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Window soft-argmax on a similarity map.

        Args:
            similarity: Tensor (H, W) with similarity scores.
            window_size: Odd int, side length of local window centered on hard argmax.

        Returns:
            pred_y, pred_x: scalar tensors (float) in similarity-map coordinates.
        """
        assert similarity.ndim == 2, "similarity must be a 2D tensor (H, W)"
        assert window_size % 2 == 1, "window_size must be odd"

        H, W = similarity.shape
        device = similarity.device

        # 1) hard argmax to get window center
        max_idx = torch.argmax(similarity)  # scalar
        max_y = (max_idx // W).item()
        max_x = (max_idx % W).item()

        # 2) define local window bounds (clamped)
        half = window_size // 2
        y0 = max(0, max_y - half)
        y1 = min(H, max_y + half + 1)  # exclusive
        x0 = max(0, max_x - half)
        x1 = min(W, max_x + half + 1)  # exclusive

        window = similarity[y0:y1, x0:x1]  # (h, w)

        # 3) softmax over window values (flattened)
        weights = F.softmax(window.flatten(), dim=0)

        # 4) coordinates grid for the window, converted to global coords
        h, w = window.shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )
        ys = (ys + y0).flatten().float()
        xs = (xs + x0).flatten().float()

        # 5) weighted average -> soft-argmax result
        pred_y = torch.sum(weights * ys)
        pred_x = torch.sum(weights * xs)

        return pred_y, pred_x
