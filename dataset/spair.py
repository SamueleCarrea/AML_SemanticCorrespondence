from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PairAnnotation:
    """Container for the metadata of a single pair."""

    pair_id: str
    category: str
    annotation_path: Path


class SPairDataset(Dataset):
    """PyTorch dataset for the SPair-71k benchmark.

    The dataset expects the following structure under ``root``::

        root/
            JPEGImages/
            PairAnnotation/<class_name>/*.json
            ImageSets/main/{train,val,test}.txt
            symmetry.txt (optional)

    Each entry in the split file corresponds to a JSON annotation containing
    the image names, bounding boxes and keypoints for a source-target pair.
    """

    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "train",
        long_side: int = 480,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.long_side = long_side
        self.normalize = normalize

        split_file = self.root / "ImageSets" / "main" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        self.pairs: List[PairAnnotation] = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                category = line.split("_")[0]
                ann_path = self.root / "PairAnnotation" / category / f"{line}.json"
                if not ann_path.exists():
                    raise FileNotFoundError(f"Annotation not found: {ann_path}")
                self.pairs.append(PairAnnotation(pair_id=line, category=category, annotation_path=ann_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        entry = self.pairs[index]
        with open(entry.annotation_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        src_im = self._load_image(ann.get("src_imname"))
        tgt_im = self._load_image(ann.get("trg_imname"))

        src_tensor, src_scale, src_size = self._resize_and_normalize(src_im)
        tgt_tensor, tgt_scale, tgt_size = self._resize_and_normalize(tgt_im)

        src_kps, src_vis = self._load_keypoints(ann.get("src_kps", []), src_scale)
        tgt_kps, tgt_vis = self._load_keypoints(ann.get("trg_kps", []), tgt_scale)

        # Ensure the same number of keypoints between source and target
        num_kps = min(src_kps.shape[0], tgt_kps.shape[0])
        src_kps = src_kps[:num_kps]
        tgt_kps = tgt_kps[:num_kps]
        src_vis = src_vis[:num_kps]
        tgt_vis = tgt_vis[:num_kps]
        valid_mask = src_vis & tgt_vis

        sample: Dict[str, torch.Tensor | str | Tuple[int, int]] = {
            "src_img": src_tensor,
            "tgt_img": tgt_tensor,
            "src_kps": src_kps,
            "tgt_kps": tgt_kps,
            "valid_mask": valid_mask,
            "category": entry.category,
            "pair_id": entry.pair_id,
            "src_scale": torch.tensor(src_scale, dtype=torch.float32),
            "tgt_scale": torch.tensor(tgt_scale, dtype=torch.float32),
            "src_orig_size": torch.tensor(src_size, dtype=torch.int64),
            "tgt_orig_size": torch.tensor(tgt_size, dtype=torch.int64),
        }

        if "src_bbox" in ann:
            sample["src_bbox"] = torch.tensor(ann["src_bbox"], dtype=torch.float32) * src_scale
        if "trg_bbox" in ann:
            sample["tgt_bbox"] = torch.tensor(ann["trg_bbox"], dtype=torch.float32) * tgt_scale

        return sample

    def _load_image(self, image_name: Optional[str]) -> Image.Image:
        if image_name is None:
            raise ValueError("Annotation missing image name")

        img_path = self.root / "JPEGImages" / image_name
        if not img_path.exists():
            # Try adding a default jpg extension when missing
            if img_path.suffix == "":
                candidate = img_path.with_suffix(".jpg")
                if candidate.exists():
                    img_path = candidate
            # Fallback: try png
            if not img_path.exists():
                candidate = img_path.with_suffix(".png")
                if candidate.exists():
                    img_path = candidate
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def _resize_and_normalize(self, image: Image.Image) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        orig_w, orig_h = image.size
        scale = float(self.long_side) / float(max(orig_w, orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        resized = F.resize(image, size=[new_h, new_w])
        tensor = F.to_tensor(resized)
        if self.normalize:
            tensor = F.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, scale, (orig_h, orig_w)

    def _load_keypoints(
        self, kps: List[List[float]], scale: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(kps) == 0:
            return torch.zeros((0, 2), dtype=torch.float32), torch.zeros(0, dtype=torch.bool)

        coords = torch.tensor([[kp[0], kp[1]] for kp in kps], dtype=torch.float32) * scale
        visibility = torch.tensor([kp[2] if len(kp) > 2 else 1 for kp in kps], dtype=torch.bool)
        return coords, visibility
