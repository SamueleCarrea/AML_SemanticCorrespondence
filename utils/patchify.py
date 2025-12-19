import math
from typing import Iterable, Tuple

import torch


def compute_feature_map_shape(
    image_shape: Tuple[int, int], stride: int, padding: int = 0
) -> Tuple[int, int]:
    """
    Calculate the spatial shape of a feature map given an input image.

    Args:
        image_shape: ``(height, width)`` of the input image.
        stride: stride or patch size used by the encoder.
        padding: optional symmetric padding (pixels) applied before encoding.
    """
    h, w = image_shape
    h_feat = math.floor((h + 2 * padding) / stride)
    w_feat = math.floor((w + 2 * padding) / stride)
    return h_feat, w_feat


def image_to_feature_points(
    points: torch.Tensor, stride: int, padding: int = 0, clamp: bool = True
) -> torch.Tensor:
    """
    Map image-space coordinates to feature-map indices.

    Args:
        points: tensor of ``(..., 2)`` with ``(x, y)`` pixel coordinates.
        stride: feature map stride.
        padding: symmetric padding applied before the encoder.
        clamp: if ``True`` clamp indices to the valid feature range.
    """
    coords = (points + padding).float() / float(stride)
    coords = coords.floor().long()
    if clamp:
        coords = torch.clamp(coords, min=0)
    return coords


def image_to_feature_boxes(
    boxes: torch.Tensor, stride: int, padding: int = 0, clamp: bool = True
) -> torch.Tensor:
    """
    Convert bounding boxes from image pixels to feature-map indices.

    Args:
        boxes: tensor of shape ``(..., 4)`` with ``(x1, y1, x2, y2)``.
        stride: feature map stride.
        padding: symmetric padding applied before the encoder.
        clamp: if ``True`` clamp indices to valid feature range.
    """
    xy1 = image_to_feature_points(boxes[..., :2], stride, padding, clamp=False)
    xy2 = image_to_feature_points(boxes[..., 2:], stride, padding, clamp=False)
    boxes_feat = torch.stack([xy1[..., 0], xy1[..., 1], xy2[..., 0], xy2[..., 1]], dim=-1)
    if clamp:
        boxes_feat = torch.clamp(boxes_feat, min=0)
    return boxes_feat


def feature_to_image_points(
    feature_points: torch.Tensor, stride: int, padding: int = 0, center: bool = True
) -> torch.Tensor:
    """
    Project feature-map indices back to image pixel coordinates.

    Args:
        feature_points: tensor of ``(..., 2)`` with feature indices.
        stride: feature map stride.
        padding: symmetric padding applied before the encoder.
        center: if ``True`` return the patch center; otherwise the top-left corner.
    """
    offset = 0.5 if center else 0.0
    coords = (feature_points.float() + offset) * float(stride) - padding
    return coords


def patch_grid(
    image_shape: Tuple[int, int], stride: int, padding: int = 0
) -> torch.Tensor:
    """
    Build a grid of patch centers aligned to a feature map.

    Returns:
        Tensor of shape ``(H_feat, W_feat, 2)`` with ``(x, y)`` coordinates.
    """
    h_feat, w_feat = compute_feature_map_shape(image_shape, stride, padding)
    ys = torch.arange(h_feat)
    xs = torch.arange(w_feat)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return feature_to_image_points(grid, stride=stride, padding=padding, center=True)
