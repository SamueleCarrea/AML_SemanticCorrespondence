from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import json

import numpy as np
from PIL import Image, ImageDraw

from utils.vis import to_pil_image


@dataclass
class PairSample:
    """Container for a source/target pair with keypoints and matches."""

    source_image: Image.Image
    target_image: Image.Image
    source_keypoints: np.ndarray
    target_keypoints: np.ndarray
    matches: List[Tuple[int, int]]
    name: str = "pair"


def _grid_background(size: int, rng: np.random.Generator) -> Image.Image:
    image = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    spacing = max(16, size // 12)
    for pos in range(0, size, spacing):
        shade = 180 + int(rng.random() * 40)
        draw.line([(pos, 0), (pos, size)], fill=(shade, shade, shade), width=1)
        draw.line([(0, pos), (size, pos)], fill=(shade, shade, shade), width=1)

    for _ in range(8):
        x0, y0 = rng.integers(0, size - spacing, size=2)
        x1, y1 = x0 + rng.integers(spacing // 2, spacing), y0 + rng.integers(spacing // 2, spacing)
        color = tuple(int(c) for c in rng.integers(50, 220, size=3))
        draw.rectangle([(int(x0), int(y0)), (int(x1), int(y1))], outline=color, width=2)
    return image


def _random_keypoints(
    rng: np.random.Generator, num_keypoints: int, image_size: int, border: int = 8
) -> np.ndarray:
    return rng.uniform(border, image_size - border, size=(num_keypoints, 2)).astype(float)


def generate_synthetic_pair(
    seed: int = 0,
    name: str | None = None,
    num_keypoints: int = 8,
    image_size: int = 192,
    offset: Tuple[float, float] | None = None,
    noise: float = 1.5,
) -> PairSample:
    """
    Generate a lightweight synthetic pair for debugging and examples.

    Args:
        seed: Seed for random generator.
        name: Optional name for the pair.
        num_keypoints: Number of keypoints to sample.
        image_size: Size of the square images.
        offset: Deterministic translation applied to target keypoints. If None, sampled randomly.
        noise: Standard deviation of per-point noise added to the offset.
    """
    rng = np.random.default_rng(seed)
    chosen_offset = offset or tuple(rng.integers(-18, 18, size=2))
    offset_array = np.asarray(chosen_offset, dtype=float)

    source_image = _grid_background(image_size, rng)
    target_image = _grid_background(image_size, rng)

    source_keypoints = _random_keypoints(rng, num_keypoints, image_size, border=image_size // 10)
    per_point_noise = rng.normal(scale=noise, size=source_keypoints.shape)
    target_keypoints = np.clip(source_keypoints + offset_array + per_point_noise, 2, image_size - 2)

    matches = [(idx, idx) for idx in range(num_keypoints)]
    pair_name = name or f"synthetic_{seed}"

    return PairSample(
        source_image=source_image,
        target_image=target_image,
        source_keypoints=source_keypoints,
        target_keypoints=target_keypoints,
        matches=matches,
        name=pair_name,
    )


def generate_synthetic_dataset(num_samples: int, seed: int = 0) -> List[PairSample]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10_000, size=num_samples)
    return [
        generate_synthetic_pair(seed=int(sample_seed), name=f"synthetic_{i}") for i, sample_seed in enumerate(seeds)
    ]


def _load_keypoints(data: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(data, dtype=float)


def _load_matches(raw_matches: Iterable[Sequence[int]] | None, num_keypoints: int) -> List[Tuple[int, int]]:
    if raw_matches is None:
        return [(i, i) for i in range(num_keypoints)]
    return [(int(src), int(tgt)) for src, tgt in raw_matches]


def load_pair_from_npz(path: Path) -> PairSample:
    loaded = np.load(path, allow_pickle=True)
    source_image = to_pil_image(loaded["source_image"])
    target_image = to_pil_image(loaded["target_image"])
    source_keypoints = _load_keypoints(loaded["source_keypoints"])
    target_keypoints = _load_keypoints(loaded["target_keypoints"])
    matches_array = loaded["matches"] if "matches" in loaded.files else None
    matches = _load_matches(matches_array, num_keypoints=min(len(source_keypoints), len(target_keypoints)))
    name = str(loaded["name"]) if "name" in loaded.files else path.stem
    return PairSample(
        source_image=source_image,
        target_image=target_image,
        source_keypoints=source_keypoints,
        target_keypoints=target_keypoints,
        matches=matches,
        name=name,
    )


def load_pair_from_json(path: Path) -> PairSample:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    source_image_path = Path(raw["source_image"])
    target_image_path = Path(raw["target_image"])
    with Image.open(source_image_path) as src_img:
        source_image = to_pil_image(src_img.copy())
    with Image.open(target_image_path) as tgt_img:
        target_image = to_pil_image(tgt_img.copy())

    source_keypoints = _load_keypoints(raw["source_keypoints"])
    target_keypoints = _load_keypoints(raw["target_keypoints"])
    matches = _load_matches(raw.get("matches"), num_keypoints=min(len(source_keypoints), len(target_keypoints)))
    name = raw.get("name", path.stem)
    return PairSample(
        source_image=source_image,
        target_image=target_image,
        source_keypoints=source_keypoints,
        target_keypoints=target_keypoints,
        matches=matches,
        name=name,
    )


def load_pair(path: str | Path, seed: int = 0) -> PairSample:
    """
    Load a pair from disk or fall back to a synthetic pair.

    Supported formats:
      - .npz files with keys source_image, target_image, source_keypoints, target_keypoints, optional matches/name.
      - .json manifest referencing image paths and keypoint lists.
      - the literal string ``synthetic`` to generate an in-memory example.
    """
    if isinstance(path, str) and path.lower() == "synthetic":
        return generate_synthetic_pair(seed=seed, name="synthetic_pair")

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Pair file not found: {target_path}")

    if target_path.suffix.lower() == ".npz":
        return load_pair_from_npz(target_path)
    if target_path.suffix.lower() == ".json":
        return load_pair_from_json(target_path)

    raise ValueError(f"Unsupported pair format: {target_path.suffix}")
