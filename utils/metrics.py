"""Metric utilities for keypoint-based evaluation.

This module provides helpers to compute the Percentage of Correct Keypoints (PCK)
for individual images as well as dataset-level aggregations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _compute_bbox_sides(gt_kps: np.ndarray, valid_mask: np.ndarray) -> Tuple[float, float]:
    """Compute the height and width of the bounding box covering valid keypoints."""

    valid_gt = gt_kps[valid_mask]
    if valid_gt.size == 0:
        return 1.0, 1.0

    min_xy = valid_gt.min(axis=0)
    max_xy = valid_gt.max(axis=0)
    width = float(max_xy[0] - min_xy[0])
    height = float(max_xy[1] - min_xy[1])

    # Avoid degenerate boxes that would nullify the threshold.
    if width <= 0:
        width = 1.0
    if height <= 0:
        height = 1.0

    return height, width


def compute_pck(
    pred_kps: Sequence[Sequence[float]],
    gt_kps: Sequence[Sequence[float]],
    valid_mask: Optional[Sequence[bool]],
    alpha: float,
) -> Tuple[float, int]:
    """Compute the PCK for a single image.

    The correctness threshold is defined as ``T = alpha * max(h, w)``, where
    ``h`` and ``w`` are derived from the bounding box of the ground-truth
    keypoints (restricted to the valid ones).

    Args:
        pred_kps: Predicted keypoints with shape ``(num_kps, 2)``.
        gt_kps: Ground-truth keypoints with shape ``(num_kps, 2)``.
        valid_mask: Boolean mask indicating which keypoints should be evaluated.
        alpha: Threshold multiplier.

    Returns:
        A tuple ``(pck, num_valid)`` where ``pck`` is the percentage of correct
        keypoints for the provided ``alpha`` and ``num_valid`` is the count of
        valid keypoints considered in the computation.
    """

    pred_arr = np.asarray(pred_kps, dtype=float)
    gt_arr = np.asarray(gt_kps, dtype=float)

    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"pred_kps and gt_kps must share the same shape, got {pred_arr.shape} and {gt_arr.shape}."
        )

    if valid_mask is None:
        valid_mask_arr = np.ones(pred_arr.shape[0], dtype=bool)
    else:
        valid_mask_arr = np.asarray(valid_mask, dtype=bool)
        if valid_mask_arr.shape[0] != pred_arr.shape[0]:
            raise ValueError(
                f"valid_mask must align with keypoints; expected {pred_arr.shape[0]}, got {valid_mask_arr.shape[0]}."
            )

    # Exclude keypoints with non-finite coordinates to avoid propagating NaNs.
    finite_mask = np.all(np.isfinite(gt_arr), axis=1) & np.all(np.isfinite(pred_arr), axis=1)
    eval_mask = valid_mask_arr & finite_mask

    if not np.any(eval_mask):
        return 0.0, 0

    height, width = _compute_bbox_sides(gt_arr, eval_mask)
    threshold = alpha * max(height, width)

    distances = np.linalg.norm(pred_arr[eval_mask] - gt_arr[eval_mask], axis=1)
    correct = distances <= threshold
    pck_value = float(np.mean(correct))

    return pck_value, int(np.sum(eval_mask))


@dataclass
class PCKAggregator:
    """Aggregate PCK values per image, per category, and across thresholds."""

    alpha_list: Sequence[float]

    def __post_init__(self) -> None:
        self.overall: MutableMapping[float, List[float]] = {alpha: [] for alpha in self.alpha_list}
        self.per_category: Dict[str, MutableMapping[float, List[float]]] = defaultdict(
            lambda: {alpha: [] for alpha in self.alpha_list}
        )

    def update(
        self,
        pred_kps: Sequence[Sequence[float]],
        gt_kps: Sequence[Sequence[float]],
        valid_mask: Optional[Sequence[bool]],
        category: Optional[str] = None,
    ) -> Dict[float, float]:
        """Update the aggregator with a single image prediction.

        Returns a mapping from ``alpha`` to the image-level PCK values that were
        computed.
        """

        image_scores: Dict[float, float] = {}
        for alpha in self.alpha_list:
            pck_value, num_valid = compute_pck(pred_kps, gt_kps, valid_mask, alpha)
            if num_valid == 0:
                continue

            image_scores[alpha] = pck_value
            self.overall[alpha].append(pck_value)
            if category is not None:
                self.per_category[category][alpha].append(pck_value)

        return image_scores

    def summarize(self) -> Dict[str, Dict[float, float]]:
        """Return dataset-level averages per threshold and per category."""

        overall_summary = {
            alpha: float(np.mean(scores)) if len(scores) > 0 else 0.0 for alpha, scores in self.overall.items()
        }

        category_summary: Dict[str, Dict[float, float]] = {}
        for category, alpha_scores in self.per_category.items():
            category_summary[category] = {
                alpha: float(np.mean(scores)) if len(scores) > 0 else 0.0 for alpha, scores in alpha_scores.items()
            }

        return {
            "overall": overall_summary,
            "per_category": category_summary,
        }

