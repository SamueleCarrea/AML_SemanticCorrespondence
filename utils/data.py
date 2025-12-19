from __future__ import annotations

from typing import Any, Dict, List

import torch


def keypoint_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that pads variable-length keypoints and masks.

    Args:
        batch: List of dataset samples.

    Returns:
        A dictionary with stacked images and padded keypoints/masks.
    """

    src_imgs = torch.stack([item["src_img"] for item in batch])
    tgt_imgs = torch.stack([item["tgt_img"] for item in batch])

    num_kps = [item["src_kps"].shape[0] for item in batch]
    max_kps = max(num_kps) if num_kps else 0

    def _pad_tensor(t: torch.Tensor, pad_value: float = 0.0) -> torch.Tensor:
        if t.shape[0] == max_kps:
            return t
        pad_shape = (max_kps - t.shape[0], *t.shape[1:])
        padding = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
        return torch.cat([t, padding], dim=0)

    src_kps = torch.stack([_pad_tensor(item["src_kps"]) for item in batch])
    tgt_kps = torch.stack([_pad_tensor(item["tgt_kps"]) for item in batch])

    valid_mask = torch.stack(
        [
            _pad_tensor(item["valid_mask"].unsqueeze(-1), pad_value=0).squeeze(-1)
            for item in batch
        ]
    )

    collated: Dict[str, Any] = {
        "src_img": src_imgs,
        "tgt_img": tgt_imgs,
        "src_kps": src_kps,
        "tgt_kps": tgt_kps,
        "valid_mask": valid_mask.bool(),
        "category": [item["category"] for item in batch],
        "pair_id": [item["pair_id"] for item in batch],
        "num_keypoints": torch.tensor(num_kps, dtype=torch.int64),
    }

    # Propagate optional fields if present
    optional_keys = ["src_bbox", "tgt_bbox", "src_scale", "tgt_scale", "src_orig_size", "tgt_orig_size"]
    for key in optional_keys:
        if key in batch[0]:
            values = [item[key] for item in batch]
            if torch.is_tensor(values[0]):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values

    return collated
