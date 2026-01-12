"""SPair-71k Dataset Loader.

This module provides a PyTorch Dataset implementation for the SPair-71k benchmark
for semantic correspondence evaluation.

The dataset expects the following structure:
    root/
        ImageAnnotation/
            train/pairs.json
            val/pairs.json
            test/pairs.json
        JPEGImages/
            <category>/*.jpg
        keypoints/
            <category>.json (optional)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SPairDataset(Dataset):
    """PyTorch dataset for SPair-71k semantic correspondence benchmark.

    SPair-71k contains 70,958 image pairs across 18 object categories with
    manually annotated keypoint correspondences.

    Args:
        root: Path to SPair-71k dataset root directory
        split: Dataset split - 'train', 'val', or 'test'
        long_side: Resize images so max(H, W) = long_side
        normalize: Apply ImageNet normalization

    Expected directory structure:
        root/
            ImageAnnotation/
                train/pairs.json  (7,200 pairs)
                val/pairs.json    (1,800 pairs)
                test/pairs.json   (1,814 pairs)
            JPEGImages/
                aeroplane/*.jpg
                bicycle/*.jpg
                ...

    Returns:
        Dictionary with keys:
            - src_img: (3, H, W) source image tensor
            - tgt_img: (3, H, W) target image tensor
            - src_kps: (N, 2) source keypoints in (x, y) format
            - tgt_kps: (N, 2) target keypoints in (x, y) format
            - valid_mask: (N,) boolean mask for valid keypoints
            - category: str, object category name
            - pair_id: str, unique pair identifier
            - src_scale: float, resize scale factor for source
            - tgt_scale: float, resize scale factor for target
            - src_orig_size: (2,) original source image (H, W)
            - tgt_orig_size: (2,) original target image (H, W)

    Example:
        >>> dataset = SPairDataset(root='data/SPair-71k', split='test')
        >>> sample = dataset[0]
        >>> src_img = sample['src_img']  # (3, H, W)
        >>> src_kps = sample['src_kps']  # (N, 2)
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "test",
        long_side: int = 518,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.long_side = long_side
        self.normalize = normalize

        # Load annotation file
        ann_file = self.root / "ImageAnnotation" / split / "pairs.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Expected structure: {self.root}/ImageAnnotation/{split}/pairs.json"
            )

        with open(ann_file, "r", encoding="utf-8") as f:
            self.pairs = json.load(f)

        if len(self.pairs) == 0:
            raise ValueError(f"No pairs found in {ann_file}")

        print(f"✅ Loaded {len(self.pairs)} pairs from {split} split")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Tuple[int, int]]:
        """Load a single source-target image pair with keypoint annotations."""
        pair = self.pairs[index]

        # Load images
        src_img = self._load_image(pair["src_img"])
        tgt_img = self._load_image(pair["tgt_img"])

        # Resize and normalize
        src_tensor, src_scale, src_orig_size = self._resize_and_normalize(src_img)
        tgt_tensor, tgt_scale, tgt_orig_size = self._resize_and_normalize(tgt_img)

        # Load keypoints and scale them
        src_kps = torch.tensor(pair["src_kps"], dtype=torch.float32) * src_scale
        tgt_kps = torch.tensor(pair["tgt_kps"], dtype=torch.float32) * tgt_scale

        # Load visibility mask
        valid_mask = torch.tensor(pair["valid"], dtype=torch.bool)

        # Ensure consistent number of keypoints
        num_kps = min(len(src_kps), len(tgt_kps), len(valid_mask))
        src_kps = src_kps[:num_kps]
        tgt_kps = tgt_kps[:num_kps]
        valid_mask = valid_mask[:num_kps]

        sample = {
            "src_img": src_tensor,
            "tgt_img": tgt_tensor,
            "src_kps": src_kps,
            "tgt_kps": tgt_kps,
            "valid_mask": valid_mask,
            "category": pair["category"],
            "pair_id": pair["pair_id"],
            "src_scale": torch.tensor(src_scale, dtype=torch.float32),
            "tgt_scale": torch.tensor(tgt_scale, dtype=torch.float32),
            "src_orig_size": torch.tensor(src_orig_size, dtype=torch.int64),
            "tgt_orig_size": torch.tensor(tgt_orig_size, dtype=torch.int64),
        }

        # Add bounding boxes if available
        if "src_bbox" in pair:
            sample["src_bbox"] = torch.tensor(pair["src_bbox"], dtype=torch.float32) * src_scale
        if "tgt_bbox" in pair:
            sample["tgt_bbox"] = torch.tensor(pair["tgt_bbox"], dtype=torch.float32) * tgt_scale

        return sample

    def _load_image(self, image_name: str) -> Image.Image:
        """Load an image from JPEGImages directory."""
        img_path = self.root / "JPEGImages" / image_name

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Expected location: {self.root}/JPEGImages/{image_name}"
            )

        return Image.open(img_path).convert("RGB")

    def _resize_and_normalize(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Resize image maintaining aspect ratio and optionally normalize."""
        orig_w, orig_h = image.size

        # Calculate scale to resize long side to target size
        scale = float(self.long_side) / float(max(orig_w, orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # Resize
        resized = F.resize(image, size=[new_h, new_w])

        # Convert to tensor
        tensor = F.to_tensor(resized)

        # Normalize with ImageNet statistics
        if self.normalize:
            tensor = F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

        return tensor, scale, (orig_h, orig_w)


def compute_pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    image_size: Tuple[int, int],
    thresholds: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> Dict[str, float]:
    """Compute Percentage of Correct Keypoints (PCK) metric.

    A predicted keypoint is correct if:
        ||pred_kp - gt_kp||_2 <= threshold * max(H, W)

    Args:
        pred_kps: (N, 2) predicted keypoints in (x, y) format
        gt_kps: (N, 2) ground truth keypoints in (x, y) format
        image_size: (H, W) image dimensions
        thresholds: List of normalized distance thresholds (e.g., 0.10 = 10%)

    Returns:
        Dictionary mapping 'PCK@{threshold}' to accuracy value

    Example:
        >>> pred = torch.tensor([[100.0, 150.0], [200.0, 250.0]])
        >>> gt = torch.tensor([[105.0, 155.0], [195.0, 245.0]])
        >>> pck = compute_pck(pred, gt, (480, 640))
        >>> print(f"PCK@0.10: {pck['PCK@0.10']:.4f}")
    """
    H, W = image_size
    max_dim = max(H, W)

    # Compute Euclidean distances
    distances = torch.norm(pred_kps - gt_kps, dim=1)  # (N,)

    # Normalize by max image dimension
    normalized_distances = distances / max_dim

    # Compute PCK for each threshold
    results = {}
    for threshold in thresholds:
        correct = (normalized_distances <= threshold).float()
        pck = correct.mean().item()
        results[f"PCK@{threshold:.2f}"] = pck

    return results


def compute_pck_per_category(
    pred_kps_list: List[torch.Tensor],
    gt_kps_list: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    categories: List[str],
    thresholds: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> Dict[str, Dict[str, float]]:
    """Compute PCK metrics grouped by object category.

    Args:
        pred_kps_list: List of (N, 2) predicted keypoints tensors
        gt_kps_list: List of (N, 2) ground truth keypoints tensors
        image_sizes: List of (H, W) tuples
        categories: List of category names
        thresholds: List of PCK thresholds

    Returns:
        Dictionary mapping category -> {PCK@T metrics}

    Example:
        >>> results = compute_pck_per_category(preds, gts, sizes, cats)
        >>> print(results['cat']['PCK@0.10'])
    """
    from collections import defaultdict

    category_results = defaultdict(lambda: defaultdict(list))

    for pred, gt, size, cat in zip(pred_kps_list, gt_kps_list, image_sizes, categories):
        pck = compute_pck(pred, gt, size, thresholds)
        for metric, value in pck.items():
            category_results[cat][metric].append(value)

    # Average per category
    final_results = {}
    for cat, metrics in category_results.items():
        final_results[cat] = {
            metric: sum(values) / len(values) for metric, values in metrics.items()
        }

    return final_results
