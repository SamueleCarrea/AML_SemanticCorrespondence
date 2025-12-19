import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from utils.metrics import PCKAggregator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline model with PCK metrics.")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file.")
    parser.add_argument("--backbone", type=str, default=None, help="Optional backbone override.")
    parser.add_argument(
        "--alpha_list",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
        help="List of alpha thresholds for PCK computation.",
    )
    parser.add_argument("--output", type=str, default="eval_report", help="Base name for output report.")
    return parser.parse_args()


def dummy_load_dataloader(split: str) -> List[Dict[str, Any]]:
    """Placeholder dataloader to be replaced with project-specific loading.

    Returns a list of dictionaries with keys: 'pred_kps', 'gt_kps', 'valid_mask', 'category'.
    """

    rng = np.random.default_rng(0)
    data = []
    for idx in range(5):
        num_kps = 10
        gt = rng.uniform(0, 100, size=(num_kps, 2))
        pred = gt + rng.normal(0, 5, size=(num_kps, 2))
        mask = rng.random(num_kps) > 0.1
        category = f"cat_{idx % 2}"
        data.append({"pred_kps": pred, "gt_kps": gt, "valid_mask": mask, "category": category})
    return data


def run_baseline_inference(config_path: str, backbone: str | None) -> List[Dict[str, Any]]:
    """Run the baseline model and return predictions and ground truth.

    This function is a stand-in for the project's model inference pipeline. It should be
    replaced with the actual inference implementation that returns iterable samples.
    """

    _ = config_path, backbone
    return dummy_load_dataloader(split="val")


def save_reports(base_path: Path, summary: Dict[str, Dict[float, float]]) -> None:
    """Save the evaluation summary to JSON and CSV files."""

    base_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = base_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    csv_path = base_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8") as f:
        # CSV header
        headers = ["category"] + [f"alpha_{alpha}" for alpha in summary["overall"].keys()]
        f.write(",".join(headers) + "\n")

        # Overall row
        overall_row = ["overall"] + [str(summary["overall"][alpha]) for alpha in summary["overall"].keys()]
        f.write(",".join(overall_row) + "\n")

        for category, scores in summary["per_category"].items():
            row = [category] + [str(scores.get(alpha, 0.0)) for alpha in summary["overall"].keys()]
            f.write(",".join(row) + "\n")


def main() -> None:
    args = parse_args()
    samples = run_baseline_inference(args.config, args.backbone)

    aggregator = PCKAggregator(alpha_list=args.alpha_list)

    for sample in samples:
        aggregator.update(
            pred_kps=sample["pred_kps"],
            gt_kps=sample["gt_kps"],
            valid_mask=sample.get("valid_mask"),
            category=sample.get("category"),
        )

    summary = aggregator.summarize()
    print(json.dumps(summary, indent=2))

    output_base = Path(args.output)
    save_reports(output_base, summary)


if __name__ == "__main__":
    main()
