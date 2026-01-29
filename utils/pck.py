
from typing import Dict, List, Optional, Tuple
import torch


def compute_pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    bbox_size: Optional[Tuple[int, int]] = None,
    thresholds: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """Compute Percentage of Correct Keypoints (PCK) metric.

    A predicted keypoint is correct if:
        ||pred_kp - gt_kp||_2 <= threshold * max(H, W)

    Args:
        pred_kps: (N, 2) predicted keypoints in (x, y) format
        gt_kps: (N, 2) ground truth keypoints in (x, y) format
        image_size: (H, W) image dimensions (if bbox_size not provided)
        bbox_size: (H, W) bounding box dimensions (preferred)
        thresholds: List of normalized distance thresholds (e.g., 0.10 = 10%)

    Returns:
        Dictionary mapping 'PCK@{threshold}' to tuple (valid_mask, accuracy):
        - valid_mask: (N,) binary tensor indicating which keypoints are correct(1 if pred within threshold, 0 otherwise). Can be used to compute per-keypoint PCK.
        - accuracy: float, mean accuracy across all keypoints for this threshold. Can be used to compute per-image PCK.
    Example:
        >>> pred = torch.tensor([[100.0, 150.0], [200.0, 250.0]])
        >>> gt = torch.tensor([[105.0, 155.0], [195.0, 245.0]])
        >>> pck = compute_pck(pred, gt, image_size=(480, 640))
        >>> valid_mask, accuracy = pck['PCK@0.10']
        >>> print(f"PCK@0.10: {accuracy:.4f}")
    """
    # Note: Both image_size and bbox_size expected in (H, W) format
    if bbox_size is not None:
        H, W = bbox_size
        max_dim = max(H, W)
    elif image_size is not None:
        H, W = image_size
        max_dim = max(H, W)
    else:
        raise ValueError("Either image_size or bbox_size must be provided")

    # Compute Euclidean distances
    distances = torch.norm(pred_kps - gt_kps, dim=1)  # (N,)

    # Normalize by max dimension
    normalized_distances = distances / max_dim

    # Compute PCK for each threshold
    results = {}
    for threshold in thresholds:
        correct = (normalized_distances <= threshold).float()
        results[f"PCK@{threshold:.2f}"] = (correct, correct.mean().item())

    return results
