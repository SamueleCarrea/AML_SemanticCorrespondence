
from typing import Dict, List, Optional, Tuple
import torch


def compute_pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    bbox_size: Optional[Tuple[int, int]] = None,
    thresholds: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> Dict[str, torch.Tensor]:
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
        Dictionary mapping 'PCK@{threshold}' to accuracy value
        and the number of keypoints evaluated
    Example:
        >>> pred = torch.tensor([[100.0, 150.0], [200.0, 250.0]])
        >>> gt = torch.tensor([[105.0, 155.0], [195.0, 245.0]])
        >>> pck = compute_pck(pred, gt, image_size=(480, 640))
        >>> print(f"PCK@0.10: {pck['PCK@0.10']:.4f}")
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
        results[f"PCK@{threshold:.2f}"] = correct

    return results


# def compute_pck_per_category(
#     pred_kps_list: List[torch.Tensor],
#     gt_kps_list: List[torch.Tensor],
#     image_sizes: List[Tuple[int, int]],
#     categories: List[str],
#     thresholds: List[float] = [0.05, 0.10, 0.15, 0.20],
# ) -> Dict[str, Dict[str, float]]:
#     """Compute PCK metrics grouped by object category.

#     Args:
#         pred_kps_list: List of (N, 2) predicted keypoints tensors
#         gt_kps_list: List of (N, 2) ground truth keypoints tensors
#         image_sizes: List of (H, W) tuples
#         categories: List of category names
#         thresholds: List of PCK thresholds

#     Returns:
#         Dictionary mapping category -> {PCK@T metrics}

#     Example:
#         >>> results = compute_pck_per_category(preds, gts, sizes, cats)
#         >>> print(results['cat']['PCK@0.10'])
#     """
#     from collections import defaultdict

#     category_results = defaultdict(lambda: defaultdict(list))

#     for pred, gt, size, cat in zip(pred_kps_list, gt_kps_list, image_sizes, categories):
#         pck = compute_pck(pred, gt, image_size=size, thresholds=thresholds)
#         for metric, value in pck.items():
#             category_results[cat][metric].append(value)

#     # Average per category
#     final_results = {}
#     for cat, metrics in category_results.items():
#         final_results[cat] = {
#             metric: sum(values) / len(values) for metric, values in metrics.items()
#         }

#     return final_results

# def compute_pck_per_keypoint(
#     pred_kps: torch.Tensor,
#     gt_kps: torch.Tensor,
#     image_size: tuple,
#     thresholds: List[float] = [0.05, 0.10, 0.15, 0.20]
# ) -> Dict[int, Dict[str, float]]:
#     """
#     Compute PCK per individual keypoint.

#     Returns:
#         Dictionary mapping keypoint_id -> {PCK@T metrics}
#     """
#     H, W = image_size
#     max_dim = max(H, W)

#     N = pred_kps.shape[0]
#     distances = torch.norm(pred_kps - gt_kps, dim=1) / max_dim

#     results = {}
#     for i in range(N):
#         kp_results = {}
#         for t in thresholds:
#             correct = (distances[i] <= t).float().item()
#             kp_results[f'PCK@{t:.2f}'] = correct
#         results[i] = kp_results

#     return results
