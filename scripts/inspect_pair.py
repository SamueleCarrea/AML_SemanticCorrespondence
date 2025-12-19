from __future__ import annotations

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import PairSample, load_pair
from utils.vis import draw_correspondences


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a correspondence pair with keypoints and matches.")
    parser.add_argument(
        "--pair",
        type=str,
        default="synthetic",
        help="Path to a .npz/.json pair file or the literal 'synthetic' to generate a toy example.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/debug"),
        help="Directory where the debug visualization will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used when generating a synthetic pair.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the visualization with matplotlib (useful locally).",
    )
    return parser.parse_args()


def _describe_pair(pair: PairSample) -> str:
    num_matches = len(pair.matches)
    return (
        f"Pair '{pair.name}': {pair.source_image.size} -> {pair.target_image.size} | "
        f"{len(pair.source_keypoints)} source kps, {len(pair.target_keypoints)} target kps, {num_matches} matches"
    )


def main() -> None:
    args = _parse_args()
    pair = load_pair(args.pair, seed=args.seed)
    print(_describe_pair(pair))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    debug_image = draw_correspondences(
        pair.source_image,
        pair.target_image,
        pair.source_keypoints,
        pair.target_keypoints,
        matches=pair.matches,
        annotate=False,
    )

    base_name = Path(args.pair).stem if args.pair != "synthetic" else pair.name
    output_path = args.output_dir / f"{base_name}_debug.png"
    debug_image.save(output_path)
    print(f"Saved debug image to {output_path}")

    if args.show:
        plt.figure(figsize=(10, 6))
        plt.imshow(debug_image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
