"""SPair-71k Dataset Loader.

This module provides a PyTorch Dataset implementation for the SPair-71k benchmark
for semantic correspondence evaluation.

The dataset expects the following structure:
    root/
        PairAnnotation/
            test/*.json
            val/*.json
            trn/*.json
        ImageAnnotation/
            <category>/*.json
        JPEGImages/
            <category>/*.jpg
        Layout/
            large/
                trn.txt
                val.txt
                test.txt
            small/
                trn.txt
                val.txt
                test.txt
        Segmentation/
            <category>/*.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        size: Split size - 'large' (70,958 pairs) or 'small' (14,160 pairs)
        long_side: Resize images so max(H, W) = long_side
        normalize: Apply ImageNet normalization
        load_segmentation: Load segmentation masks from Segmentation/ directory

    Expected directory structure:
        root/
            Layout/
                large/
                    trn.txt      (53,340 pairs)
                    val.txt      (5,384 pairs)
                    test.txt     (12,234 pairs)
                small/
                    trn.txt      (10,652 pairs)
                    val.txt      (1,070 pairs)
                    test.txt     (2,438 pairs)
            PairAnnotation/
                test/*.json      (annotations organized by split)
                val/*.json
                trn/*.json
            JPEGImages/
                <category>/*.jpg
            Segmentation/
                <category>/*.png

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
            - src_bbox: (4,) source bounding box [xmin, ymin, xmax, ymax]
            - tgt_bbox: (4,) target bounding box [xmin, ymin, xmax, ymax]
            - src_seg: (H, W) source segmentation mask (if load_segmentation=True)
            - tgt_seg: (H, W) target segmentation mask (if load_segmentation=True)

    Example:
        >>> dataset = SPairDataset(root='data/SPair-71k', split='test', size='large')
        >>> sample = dataset[0]
        >>> src_img = sample['src_img']  # (3, H, W)
        >>> src_kps = sample['src_kps']  # (N, 2)
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "test",
        size: str = "large",
        long_side: int = 518,
        normalize: bool = True,
        load_segmentation: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.size = size
        self.long_side = long_side
        self.normalize = normalize
        self.load_segmentation = load_segmentation

        # Validate split and size
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        if size not in ["large", "small"]:
            raise ValueError(f"Invalid size: {size}. Must be 'large' or 'small'")

        # Map split names to file names
        split_file_map = {
            "train": "trn.txt",
            "val": "val.txt",
            "test": "test.txt"
        }
        
        split_dir_map = {
            "train": "trn",
            "val": "val",
            "test": "test"
        }

        # Load pairs from TXT file
        pairs_file = self.root / "Layout" / size / split_file_map[split]
        if not pairs_file.exists():
            raise FileNotFoundError(
                f"Pairs file not found: {pairs_file}\n"
                f"Expected structure: {self.root}/Layout/{size}/{split_file_map[split]}"
            )

        # Parse TXT file: format is "pair_id-src_img-trg_img:category" per line
        # Example: "009662-2007_004081-2008_000196:pottedplant"
        self.pairs = []
        with open(pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Store the full filename for JSON lookup
                full_pair_name = line  # e.g., "009662-2007_004081-2008_000196:pottedplant"
                
                # Split by ':' to separate pair info and category
                parts = line.split(":")
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid line format in {pairs_file}: {line}\n"
                        f"Expected format: 'pair_id-src_img-trg_img:category'"
                    )
                
                pair_info, category = parts
                
                # Parse pair_info: "009662-2007_004081-2008_000196"
                # Split by '-' to get [pair_id, src_img, trg_img]
                pair_parts = pair_info.split("-")
                if len(pair_parts) != 3:
                    raise ValueError(
                        f"Invalid pair info format in {pairs_file}: {pair_info}\n"
                        f"Expected format: 'pair_id-src_img-trg_img'"
                    )
                
                pair_id, src_img, trg_img = pair_parts
                
                self.pairs.append({
                    "category": category,
                    "src_image": f"{category}/{src_img}.jpg",
                    "trg_image": f"{category}/{trg_img}.jpg",
                    "pair_filename": f"{full_pair_name}.json",  # Filename with .json extension
                    "split_dir": split_dir_map[split]  # Directory name for annotations
                })
        
        if len(self.pairs) == 0:
            raise ValueError(f"No pairs found in {pairs_file}")

        print(f"✅ Loaded {len(self.pairs)} pairs from {split} split ({size})")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair_annotation(self, pair_info: Dict) -> Dict:
        """Load pair annotation JSON file."""
        category = pair_info["category"]
        pair_filename = pair_info["pair_filename"]
        split_dir = pair_info["split_dir"]
        
        # Try category subdirectory first (standard structure)
        ann_path = self.root / "PairAnnotation" / split_dir / category / pair_filename
        
        # If not found, try flat structure
        if not ann_path.exists():
            ann_path = self.root / "PairAnnotation" / split_dir / pair_filename
    
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Pair annotation not found.\n"
                f"Tried:\n"
                f"  1. {self.root / 'PairAnnotation' / split_dir / category / pair_filename}\n"
                f"  2. {self.root / 'PairAnnotation' / split_dir / pair_filename}"
            )
    
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from JPEGImages directory."""
        img_path = self.root / "JPEGImages" / image_path

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"Expected location: {self.root}/JPEGImages/{image_path}"
            )

        return Image.open(img_path).convert("RGB")

    def _load_segmentation(
        self, image_path: str, orig_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Load and resize segmentation mask."""
        # Replace .jpg with .png and change directory
        seg_path = Path(image_path).with_suffix('.png')
        seg_full_path = self.root / "Segmentation" / seg_path
        
        if not seg_full_path.exists():
            # Return empty mask if segmentation not found
            return torch.zeros(target_size, dtype=torch.uint8)
        
        seg_img = Image.open(seg_full_path)
        # Resize to match the resized image
        seg_resized = F.resize(seg_img, size=list(target_size), interpolation=F.InterpolationMode.NEAREST)
        return F.pil_to_tensor(seg_resized).squeeze(0)

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

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | Tuple[int, int]]:
        """Load a single source-target image pair with keypoint annotations."""
        pair_info = self.pairs[index]
        
        src_img_path = pair_info['src_image']  # e.g., "aeroplane/2008_000033.jpg"
        tgt_img_path = pair_info['trg_image']  # e.g., "aeroplane/2008_000042.jpg"
        category = pair_info['category']
        
        # Load pair annotation
        pair_data = self._load_pair_annotation(pair_info)

        # Load images
        src_img = self._load_image(src_img_path)
        tgt_img = self._load_image(tgt_img_path)

        # Resize and normalize
        src_tensor, src_scale, src_orig_size = self._resize_and_normalize(src_img)
        tgt_tensor, tgt_scale, tgt_orig_size = self._resize_and_normalize(tgt_img)

        # Load keypoints and scale them
        src_kps = torch.tensor(pair_data["src_kps"], dtype=torch.float32)
        tgt_kps = torch.tensor(pair_data["trg_kps"], dtype=torch.float32)
        
        # Scale keypoints
        src_kps = src_kps * src_scale
        tgt_kps = tgt_kps * tgt_scale

        # All keypoints are valid by default (SPair-71k only includes valid correspondences)
        num_kps = len(src_kps)
        valid_mask = torch.ones(num_kps, dtype=torch.bool)

        # Generate pair ID
        src_name = Path(src_img_path).stem
        tgt_name = Path(tgt_img_path).stem
        pair_id = f"{category}:{src_name}-{tgt_name}"

        sample = {
            "src_img": src_tensor,
            "tgt_img": tgt_tensor,
            "src_kps": src_kps,
            "tgt_kps": tgt_kps,
            "valid_mask": valid_mask,
            "category": category,
            "pair_id": pair_id,
            "src_scale": torch.tensor(src_scale, dtype=torch.float32),
            "tgt_scale": torch.tensor(tgt_scale, dtype=torch.float32),
            "src_orig_size": torch.tensor(src_orig_size, dtype=torch.int64),
            "tgt_orig_size": torch.tensor(tgt_orig_size, dtype=torch.int64),
        }

        # Add bounding boxes (scaled) - format: [xmin, ymin, xmax, ymax]
        if "src_bndbox" in pair_data:
            bbox = pair_data["src_bndbox"]
            sample["src_bbox"] = torch.tensor(bbox, dtype=torch.float32) * src_scale
            
        if "trg_bndbox" in pair_data:
            bbox = pair_data["trg_bndbox"]
            sample["tgt_bbox"] = torch.tensor(bbox, dtype=torch.float32) * tgt_scale

        # Load segmentation masks if requested
        if self.load_segmentation:
            src_seg = self._load_segmentation(src_img_path, src_orig_size, (src_tensor.shape[1], src_tensor.shape[2]))
            tgt_seg = self._load_segmentation(tgt_img_path, tgt_orig_size, (tgt_tensor.shape[1], tgt_tensor.shape[2]))
            sample["src_seg"] = src_seg
            sample["tgt_seg"] = tgt_seg

        return sample


def compute_pck(
    pred_kps: torch.Tensor,
    gt_kps: torch.Tensor,
    image_size: Optional[Tuple[int, int]] = None,
    bbox_size: Optional[Tuple[int, int]] = None,
    thresholds: List[float] = [0.05, 0.10, 0.15, 0.20],
) -> Dict[str, float]:
    """Compute Percentage of Correct Keypoints (PCK) metric.

    A predicted keypoint is correct if:
        ||pred_kp - gt_kp||_2 <= threshold * max(H, W)

    Args:
        pred_kps: (N, 2) predicted keypoints in (x, y) format
        gt_kps: (N, 2) ground truth keypoints in (x, y) format
        image_size: (H, W) image dimensions (if bbox_size not provided)
        bbox_size: (W, H) bounding box dimensions (preferred)
        thresholds: List of normalized distance thresholds (e.g., 0.10 = 10%)

    Returns:
        Dictionary mapping 'PCK@{threshold}' to accuracy value

    Example:
        >>> pred = torch.tensor([[100.0, 150.0], [200.0, 250.0]])
        >>> gt = torch.tensor([[105.0, 155.0], [195.0, 245.0]])
        >>> pck = compute_pck(pred, gt, image_size=(480, 640))
        >>> print(f"PCK@0.10: {pck['PCK@0.10']:.4f}")
    """
    if bbox_size is not None:
        W, H = bbox_size
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
        pck = compute_pck(pred, gt, image_size=size, thresholds=thresholds)
        for metric, value in pck.items():
            category_results[cat][metric].append(value)

    # Average per category
    final_results = {}
    for cat, metrics in category_results.items():
        final_results[cat] = {
            metric: sum(values) / len(values) for metric, values in metrics.items()
        }

    return final_results
