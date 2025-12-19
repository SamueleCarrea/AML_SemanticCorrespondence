"""Evaluation metrics for semantic correspondence."""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    img_size: Tuple[int, int],
    alpha: float = 0.1,
) -> float:
    """Compute Percentage of Correct Keypoints (PCK).
    
    PCK measures the percentage of predicted keypoints that are within
    a threshold distance from the ground truth keypoints.
    
    Args:
        pred_kps: Predicted keypoints [N, 2] in (x, y) format
        gt_kps: Ground truth keypoints [N, 2] in (x, y) format
        img_size: Image size (width, height) or (height, width)
        alpha: Threshold as a fraction of image size (default: 0.1)
        
    Returns:
        PCK score as a percentage
    """
    if len(pred_kps) == 0 or len(gt_kps) == 0:
        return 0.0
    
    # Ensure same number of keypoints
    n_kps = min(len(pred_kps), len(gt_kps))
    pred_kps = pred_kps[:n_kps]
    gt_kps = gt_kps[:n_kps]
    
    # Compute distances
    distances = torch.norm(pred_kps - gt_kps, dim=1)
    
    # Compute threshold based on image size
    # Use max dimension of image
    img_size_max = max(img_size[0], img_size[1])
    threshold = alpha * img_size_max
    
    # Compute PCK
    correct = (distances <= threshold).float()
    pck = correct.mean().item() * 100.0
    
    return pck


def compute_pck_multiple_thresholds(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    img_size: Tuple[int, int],
    alphas: List[float] = [0.05, 0.1, 0.15],
) -> Dict[str, float]:
    """Compute PCK at multiple thresholds.
    
    Args:
        pred_kps: Predicted keypoints [N, 2]
        gt_kps: Ground truth keypoints [N, 2]
        img_size: Image size (width, height)
        alphas: List of alpha values for different thresholds
        
    Returns:
        Dictionary with PCK scores at different thresholds
    """
    results = {}
    for alpha in alphas:
        pck = compute_pck(pred_kps, gt_kps, img_size, alpha=alpha)
        results[f'pck@{alpha}'] = pck
    
    return results


class PCKMetric:
    """PCK metric accumulator for evaluation."""
    
    def __init__(self, alphas: List[float] = [0.05, 0.1, 0.15]):
        """Initialize PCK metric.
        
        Args:
            alphas: List of alpha values for different thresholds
        """
        self.alphas = alphas
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.scores = {f'pck@{alpha}': [] for alpha in self.alphas}
        self.n_samples = 0
    
    def update(
        self,
        pred_kps: torch.Tensor,
        gt_kps: torch.Tensor,
        img_size: Tuple[int, int],
    ):
        """Update metrics with a new sample.
        
        Args:
            pred_kps: Predicted keypoints [N, 2]
            gt_kps: Ground truth keypoints [N, 2]
            img_size: Image size (width, height)
        """
        results = compute_pck_multiple_thresholds(
            pred_kps, gt_kps, img_size, alphas=self.alphas
        )
        
        for key, value in results.items():
            self.scores[key].append(value)
        
        self.n_samples += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics.
        
        Returns:
            Dictionary with average PCK scores
        """
        avg_scores = {}
        for key, values in self.scores.items():
            if len(values) > 0:
                avg_scores[key] = np.mean(values)
            else:
                avg_scores[key] = 0.0
        
        return avg_scores
    
    def get_summary(self) -> str:
        """Get summary string of metrics.
        
        Returns:
            Summary string
        """
        avg_scores = self.compute()
        summary = f"Evaluated {self.n_samples} samples\n"
        for key, value in sorted(avg_scores.items()):
            summary += f"  {key}: {value:.2f}%\n"
        return summary
