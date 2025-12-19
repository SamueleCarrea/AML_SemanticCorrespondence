from __future__ import annotations

import importlib
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    """
    Convert a Pillow image, numpy array, or torch tensor to a Pillow RGB image.

    Args:
        image: Input image as PIL.Image.Image, numpy array (H, W, 3) or (3, H, W),
            grayscale array (H, W), or torch tensor with matching shapes.

    Returns:
        A Pillow Image in RGB mode.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    array: np.ndarray
    if isinstance(image, np.ndarray):
        array = image
    else:
        torch_module = None
        if importlib.util.find_spec("torch") is not None:
            import torch as torch_module  # type: ignore
        if torch_module is not None and isinstance(image, torch_module.Tensor):  # type: ignore[attr-defined]
            array = image.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)

    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = np.clip(array, 0.0, 1.0)
            array = (array * 255).astype(np.uint8)
        else:
            array = np.clip(array, 0, 255).astype(np.uint8)

    return Image.fromarray(array, mode="RGB")


def _normalize_keypoints(keypoints: Sequence[Sequence[float]]) -> np.ndarray:
    if isinstance(keypoints, np.ndarray):
        return keypoints.astype(float)
    return np.asarray(keypoints, dtype=float)


def draw_keypoints(
    image: Image.Image | np.ndarray,
    keypoints: Sequence[Sequence[float]],
    color: Tuple[int, int, int] | str = (255, 0, 0),
    radius: int = 4,
    width: int = 2,
    annotate: bool = False,
) -> Image.Image:
    """
    Draw keypoints on top of the provided image.

    Args:
        image: Source image as PIL image or numpy array.
        keypoints: Iterable of (x, y) pairs.
        color: Circle color.
        radius: Radius of the drawn circles.
        width: Stroke width.
        annotate: Whether to write keypoint indices next to the markers.

    Returns:
        A new Pillow Image with the drawn keypoints.
    """
    keypoints_array = _normalize_keypoints(keypoints)
    result = to_pil_image(image).copy()
    draw = ImageDraw.Draw(result)

    font: ImageFont.ImageFont | None
    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for idx, (x, y) in enumerate(keypoints_array):
        bounding_box = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bounding_box, outline=color, width=width)
        if annotate:
            label_position = (x + radius + 1, y - radius - 1)
            draw.text(label_position, str(idx), fill=color, font=font)

    return result


def draw_correspondences(
    source_image: Image.Image | np.ndarray,
    target_image: Image.Image | np.ndarray,
    source_keypoints: Sequence[Sequence[float]],
    target_keypoints: Sequence[Sequence[float]],
    matches: Iterable[Tuple[int, int]] | None = None,
    keypoint_color: Tuple[int, int, int] | str = (255, 0, 0),
    line_color: Tuple[int, int, int] | str = (0, 200, 0),
    radius: int = 4,
    line_width: int = 2,
    margin: int = 16,
    annotate: bool = False,
) -> Image.Image:
    """
    Draw matched correspondences between source and target images.

    Args:
        source_image: Source image.
        target_image: Target image.
        source_keypoints: Iterable of (x, y) points in source image.
        target_keypoints: Iterable of (x, y) points in target image.
        matches: Iterable of (source_idx, target_idx) tuples. If None, assumes identity.
        keypoint_color: Circle color for keypoints.
        line_color: Color for match lines.
        radius: Radius of drawn circles.
        line_width: Width for lines and circles.
        margin: Horizontal spacing between the stacked images.
        annotate: Whether to label keypoints with their indices.

    Returns:
        A Pillow Image with both images stacked horizontally and matches drawn.
    """
    src_kp = _normalize_keypoints(source_keypoints)
    tgt_kp = _normalize_keypoints(target_keypoints)
    if matches is None:
        matches = list(zip(range(len(src_kp)), range(len(tgt_kp))))
    matches_list: List[Tuple[int, int]] = list(matches)

    src_img = draw_keypoints(
        source_image, src_kp, color=keypoint_color, radius=radius, width=line_width, annotate=annotate
    )
    tgt_img = draw_keypoints(
        target_image, tgt_kp, color=keypoint_color, radius=radius, width=line_width, annotate=annotate
    )

    src_w, src_h = src_img.size
    tgt_w, tgt_h = tgt_img.size
    canvas_h = max(src_h, tgt_h)
    canvas_w = src_w + margin + tgt_w
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))

    canvas.paste(src_img, (0, 0))
    canvas.paste(tgt_img, (src_w + margin, 0))

    draw = ImageDraw.Draw(canvas)
    x_offset = src_w + margin
    for src_idx, tgt_idx in matches_list:
        if src_idx >= len(src_kp) or tgt_idx >= len(tgt_kp):
            continue
        src_point = tuple(src_kp[src_idx])
        tgt_point = (tgt_kp[tgt_idx][0] + x_offset, tgt_kp[tgt_idx][1])
        draw.line([src_point, tgt_point], fill=line_color, width=line_width)

    return canvas
