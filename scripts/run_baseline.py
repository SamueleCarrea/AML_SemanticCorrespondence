from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import PairSample, generate_synthetic_dataset
from utils.vis import draw_correspondences


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight baseline on a synthetic subset.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of synthetic pairs to evaluate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for synthetic data.")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save a few correspondence visualizations to outputs/debug/ for inspection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/debug"),
        help="Directory used when saving debug visualizations.",
    )
    return parser.parse_args()


def estimate_offset(pair: PairSample) -> np.ndarray:
    """
    Estimate a global offset between source and target keypoints using their centroids.
    """
    return pair.target_keypoints.mean(axis=0) - pair.source_keypoints.mean(axis=0)


def predict_targets(source_keypoints: np.ndarray, offset: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Predict target positions by translating source keypoints and adding mild noise.
    """
    jitter = rng.normal(scale=0.75, size=source_keypoints.shape)
    return source_keypoints + offset + jitter


def compute_metrics(
    predicted_targets: np.ndarray, pair: PairSample, matches: List[Tuple[int, int]], threshold: float = 5.0
) -> Dict[str, float]:
    errors: List[float] = []
    correct = 0
    for src_idx, tgt_idx in matches:
        if src_idx >= len(predicted_targets) or tgt_idx >= len(pair.target_keypoints):
            continue
        pred = predicted_targets[src_idx]
        gt = pair.target_keypoints[tgt_idx]
        error = float(np.linalg.norm(pred - gt))
        errors.append(error)
        if error <= threshold:
            correct += 1

    mean_epe = float(np.mean(errors)) if errors else float("nan")
    pck = float(correct) / len(errors) if errors else 0.0
    return {"mean_epe": mean_epe, "pck@5": pck}


def save_debug(pair: PairSample, predicted_targets: np.ndarray, output_dir: Path, index: int) -> None:
    debug_image = draw_correspondences(
        pair.source_image,
        pair.target_image,
        pair.source_keypoints,
        predicted_targets,
        matches=pair.matches,
        annotate=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_path = output_dir / f"baseline_pair_{index:03d}.png"
    debug_image.save(debug_path)
    print(f"[debug] saved {debug_path}")


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    dataset = generate_synthetic_dataset(args.num_samples, seed=args.seed)
    print(f"Evaluating baseline on {len(dataset)} synthetic pairs...")

    aggregated = {"mean_epe": [], "pck@5": []}
    for idx, pair in enumerate(dataset):
        offset = estimate_offset(pair)
        predicted_targets = predict_targets(pair.source_keypoints, offset, rng)
        metrics = compute_metrics(predicted_targets, pair, pair.matches)
        aggregated["mean_epe"].append(metrics["mean_epe"])
        aggregated["pck@5"].append(metrics["pck@5"])
        print(f"[{idx:02d}] mean_epe={metrics['mean_epe']:.3f} | pck@5={metrics['pck@5']:.3f}")

        if args.save_debug and idx < 3:
            save_debug(pair, predicted_targets, args.output_dir, idx)

    summary = {key: float(np.nanmean(values)) for key, values in aggregated.items()}
    print("\nSummary (synthetic subset):")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
