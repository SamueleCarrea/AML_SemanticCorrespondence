"""Matching algorithms for establishing correspondences."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from sklearn.linear_model import RANSACRegressor


def nearest_neighbor_match(
    src_features: torch.Tensor,
    trg_features: torch.Tensor,
    mutual: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find nearest neighbor matches between source and target features.
    
    Args:
        src_features: Source features [N, D] or [B, N, D]
        trg_features: Target features [M, D] or [B, M, D]
        mutual: Whether to use mutual nearest neighbors
        
    Returns:
        src_indices: Source feature indices
        trg_indices: Matched target feature indices
    """
    # Handle batch dimension
    if src_features.dim() == 3:
        assert src_features.shape[0] == 1, "Batch size > 1 not supported"
        src_features = src_features.squeeze(0)
        trg_features = trg_features.squeeze(0)
    
    # Normalize features
    src_features = F.normalize(src_features, p=2, dim=-1)
    trg_features = F.normalize(trg_features, p=2, dim=-1)
    
    # Compute similarity matrix [N, M]
    similarity = torch.matmul(src_features, trg_features.t())
    
    # Find nearest neighbors from source to target
    src_to_trg = torch.argmax(similarity, dim=1)  # [N]
    
    if not mutual:
        src_indices = torch.arange(len(src_features), device=src_features.device)
        return src_indices, src_to_trg
    
    # Find nearest neighbors from target to source
    trg_to_src = torch.argmax(similarity, dim=0)  # [M]
    
    # Keep only mutual nearest neighbors
    src_indices = []
    trg_indices = []
    
    for i in range(len(src_features)):
        j = src_to_trg[i]
        if trg_to_src[j] == i:
            src_indices.append(i)
            trg_indices.append(j)
    
    if len(src_indices) == 0:
        # Return empty tensors if no mutual matches found
        return torch.tensor([], device=src_features.device, dtype=torch.long), \
               torch.tensor([], device=trg_features.device, dtype=torch.long)
    
    src_indices = torch.tensor(src_indices, device=src_features.device)
    trg_indices = torch.tensor(trg_indices, device=trg_features.device)
    
    return src_indices, trg_indices


def match_descriptors_spatial(
    src_desc: torch.Tensor,
    trg_desc: torch.Tensor,
    src_kps: torch.Tensor,
    trg_kps: Optional[torch.Tensor] = None,
    mutual: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match descriptors at keypoint locations.
    
    Args:
        src_desc: Source descriptors [D, H, W]
        trg_desc: Target descriptors [D, H, W]
        src_kps: Source keypoints [N, 2] in (x, y) format
        trg_kps: Optional target keypoints for evaluation
        mutual: Whether to use mutual nearest neighbors
        
    Returns:
        matched_src_kps: Matched source keypoints [K, 2]
        matched_trg_kps: Predicted target keypoints [K, 2]
    """
    D, H, W = src_desc.shape
    
    # Sample descriptors at keypoint locations
    src_kps_norm = src_kps.clone()
    src_kps_norm[:, 0] = 2.0 * src_kps[:, 0] / (W - 1) - 1.0
    src_kps_norm[:, 1] = 2.0 * src_kps[:, 1] / (H - 1) - 1.0
    
    # Reshape for grid_sample [1, N, 1, 2]
    grid = src_kps_norm.unsqueeze(0).unsqueeze(2)
    
    # Sample source descriptors [1, D, N, 1]
    src_desc_sampled = F.grid_sample(
        src_desc.unsqueeze(0),
        grid,
        mode='bilinear',
        align_corners=True
    )
    src_desc_sampled = src_desc_sampled.squeeze(3).squeeze(0).t()  # [N, D]
    
    # Reshape target descriptors to [H*W, D]
    trg_desc_flat = trg_desc.reshape(D, -1).t()  # [H*W, D]
    
    # Find matches
    src_indices, trg_indices = nearest_neighbor_match(
        src_desc_sampled,
        trg_desc_flat,
        mutual=mutual
    )
    
    if len(src_indices) == 0:
        # Return empty tensors if no matches found
        return torch.zeros((0, 2), device=src_kps.device), \
               torch.zeros((0, 2), device=src_kps.device)
    
    # Get matched source keypoints
    matched_src_kps = src_kps[src_indices]
    
    # Convert target indices to spatial coordinates
    trg_y = trg_indices // W
    trg_x = trg_indices % W
    matched_trg_kps = torch.stack([trg_x.float(), trg_y.float()], dim=1)
    
    return matched_src_kps, matched_trg_kps


def ransac_filter(
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    threshold: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Filter matches using RANSAC.
    
    Args:
        src_kps: Source keypoints [N, 2]
        trg_kps: Target keypoints [N, 2]
        threshold: RANSAC inlier threshold
        
    Returns:
        filtered_src_kps: Filtered source keypoints [K, 2]
        filtered_trg_kps: Filtered target keypoints [K, 2]
        inlier_mask: Boolean mask of inliers
    """
    if len(src_kps) < 4:
        # Need at least 4 points for affine transform
        inlier_mask = np.ones(len(src_kps), dtype=bool)
        return src_kps, trg_kps, inlier_mask
    
    # Convert to numpy
    src_np = src_kps.cpu().numpy()
    trg_np = trg_kps.cpu().numpy()
    
    try:
        # Fit affine transformation with RANSAC
        ransac = RANSACRegressor(residual_threshold=threshold, random_state=42)
        ransac.fit(src_np, trg_np)
        
        inlier_mask = ransac.inlier_mask_
        
        if inlier_mask is None or not np.any(inlier_mask):
            inlier_mask = np.ones(len(src_kps), dtype=bool)
        
        filtered_src_kps = src_kps[inlier_mask]
        filtered_trg_kps = trg_kps[inlier_mask]
        
        return filtered_src_kps, filtered_trg_kps, inlier_mask
    
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"RANSAC failed: {e}")
        inlier_mask = np.ones(len(src_kps), dtype=bool)
        return src_kps, trg_kps, inlier_mask


class Matcher:
    """Matcher class for establishing correspondences."""
    
    def __init__(
        self,
        matching_type: str = "mutual_nn",
        use_ransac: bool = False,
        ransac_threshold: float = 3.0,
    ):
        """Initialize matcher.
        
        Args:
            matching_type: Type of matching ('nn' or 'mutual_nn')
            use_ransac: Whether to use RANSAC filtering
            ransac_threshold: RANSAC inlier threshold
        """
        self.matching_type = matching_type
        self.use_ransac = use_ransac
        self.ransac_threshold = ransac_threshold
    
    def match(
        self,
        src_desc: torch.Tensor,
        trg_desc: torch.Tensor,
        src_kps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Establish correspondences between source and target.
        
        Args:
            src_desc: Source descriptors [D, H, W]
            trg_desc: Target descriptors [D, H, W]
            src_kps: Source keypoints [N, 2]
            
        Returns:
            matched_src_kps: Matched source keypoints [K, 2]
            matched_trg_kps: Predicted target keypoints [K, 2]
        """
        mutual = (self.matching_type == "mutual_nn")
        
        matched_src_kps, matched_trg_kps = match_descriptors_spatial(
            src_desc, trg_desc, src_kps, mutual=mutual
        )
        
        if self.use_ransac and len(matched_src_kps) > 0:
            matched_src_kps, matched_trg_kps, _ = ransac_filter(
                matched_src_kps,
                matched_trg_kps,
                threshold=self.ransac_threshold
            )
        
        return matched_src_kps, matched_trg_kps
