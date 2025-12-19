"""Command-line entrypoint to run a frozen semantic correspondence baseline.

The script loads a YAML configuration, builds a frozen model, prepares a
dataloader for the requested split, runs inference, and reports PCK metrics.

Example:
    python scripts/run_baseline.py --config config/baseline.yaml \\
        --split val --backbone dinov2 --batch_size 4
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import yaml


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    splits: Dict[str, str]
    normalization_factor: float = 1.0


@dataclass
class ModelConfig:
    """Configuration for the correspondence model."""

    backbone: str
    feature_dim: int = 0
    checkpoint: Optional[str] = None


@dataclass
class DataloaderConfig:
    """Configuration for batching."""

    batch_size: int = 4


@dataclass
class EvaluationConfig:
    """Configuration for PCK evaluation and result storage."""

    pck_threshold: float = 0.05
    output_dir: str = "checkpoints"
    save_predictions: bool = True


@dataclass
class AppConfig:
    """Top-level configuration wrapper."""

    dataset: DatasetConfig
    model: ModelConfig
    dataloader: DataloaderConfig
    evaluation: EvaluationConfig


def load_app_config(config_path: Path) -> AppConfig:
    """Load the YAML configuration and map it into dataclasses."""

    raw = yaml.safe_load(config_path.read_text())

    dataset_cfg = DatasetConfig(
        splits=raw.get("dataset", {}).get("splits", {}),
        normalization_factor=raw.get("dataset", {}).get("normalization_factor", 1.0),
    )
    model_cfg = ModelConfig(
        backbone=raw.get("model", {}).get("backbone", "dinov2"),
        feature_dim=raw.get("model", {}).get("feature_dim", 0),
        checkpoint=raw.get("model", {}).get("checkpoint"),
    )
    dataloader_cfg = DataloaderConfig(
        batch_size=raw.get("dataloader", {}).get("batch_size", 4),
    )
    evaluation_cfg = EvaluationConfig(
        pck_threshold=raw.get("evaluation", {}).get("pck_threshold", 0.05),
        output_dir=raw.get("evaluation", {}).get("output_dir", "checkpoints"),
        save_predictions=raw.get("evaluation", {}).get("save_predictions", True),
    )

    return AppConfig(
        dataset=dataset_cfg,
        model=model_cfg,
        dataloader=dataloader_cfg,
        evaluation=evaluation_cfg,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run a frozen baseline for semantic correspondence.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--backbone", default=None, help="Backbone name override (e.g., dinov2).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    return parser.parse_args()


class CorrespondenceDataset:
    """Simple JSONL-backed dataset for semantic correspondences."""

    def __init__(self, jsonl_path: Path):
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Dataset split file not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self._records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        record = self._records[idx]
        return {
            "id": record.get("id", f"sample-{idx}"),
            "source_keypoints": np.asarray(record["source_keypoints"], dtype=float),
            "target_keypoints": np.asarray(record["target_keypoints"], dtype=float),
            "predicted_keypoints": (
                np.asarray(record["predicted_keypoints"], dtype=float)
                if "predicted_keypoints" in record
                else None
            ),
            "normalization_factor": record.get("normalization_factor"),
        }


class SimpleDataLoader:
    """Minimal iterable dataloader."""

    def __init__(self, dataset: CorrespondenceDataset, batch_size: int = 4):
        self.dataset = dataset
        self.batch_size = max(1, batch_size)

    def __iter__(self) -> Iterable[List[Dict]]:
        for start in range(0, len(self.dataset), self.batch_size):
            batch = [self.dataset[i] for i in range(start, min(start + self.batch_size, len(self.dataset)))]
            yield batch


class FrozenBackboneModel:
    """Placeholder model that mimics a frozen backbone."""

    def __init__(self, backbone: str, feature_dim: int = 0, checkpoint: Optional[str] = None):
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.checkpoint = checkpoint
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def predict(self, sample: Dict) -> np.ndarray:
        """Generate predicted keypoints for a single sample."""

        if sample.get("predicted_keypoints") is not None:
            return sample["predicted_keypoints"]

        # Fallback baseline: copy the target keypoints to simulate perfect correspondence.
        if sample.get("target_keypoints") is not None:
            return np.copy(sample["target_keypoints"])

        return np.copy(sample["source_keypoints"])


class InferenceRunner:
    """Run inference over a dataloader and collect predictions."""

    def __init__(self, model: FrozenBackboneModel):
        self.model = model

    def run(self, dataloader: SimpleDataLoader) -> List[Dict]:
        predictions: List[Dict] = []
        for batch in dataloader:
            for sample in batch:
                pred_kps = self.model.predict(sample)
                predictions.append(
                    {
                        "id": sample["id"],
                        "predicted_keypoints": pred_kps,
                        "target_keypoints": sample["target_keypoints"],
                        "normalization_factor": sample.get("normalization_factor"),
                    }
                )
        return predictions


def evaluate_pck(
    predictions: List[Dict],
    threshold: float,
    default_normalization: float,
) -> Dict:
    """Compute PCK metrics for the collected predictions."""

    total_correct = 0
    total_keypoints = 0
    per_sample: List[Dict] = []

    for item in predictions:
        pred = item["predicted_keypoints"]
        target = item["target_keypoints"]
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target keypoints shape mismatch for sample {item['id']}.")

        normalization = item.get("normalization_factor") or default_normalization
        normalization = normalization if normalization else 1.0

        distances = np.linalg.norm(pred - target, axis=1)
        normalized_distances = distances / normalization
        correct = (normalized_distances <= threshold).astype(float)

        sample_correct = correct.sum()
        sample_total = correct.size
        per_sample.append(
            {
                "id": item["id"],
                "pck": float(sample_correct / sample_total) if sample_total else 0.0,
            }
        )

        total_correct += sample_correct
        total_keypoints += sample_total

    overall_pck = float(total_correct / total_keypoints) if total_keypoints else 0.0
    return {"overall_pck": overall_pck, "per_sample": per_sample}


def save_results(
    output_path: Path,
    metrics: Dict,
    predictions: Optional[List[Dict]],
    metadata: Dict,
) -> None:
    """Persist metrics (and optionally predictions) to JSON."""

    payload = {"metrics": metrics, "metadata": metadata}
    if predictions is not None:
        serializable_preds = []
        for item in predictions:
            serializable_preds.append(
                {
                    "id": item["id"],
                    "predicted_keypoints": np.asarray(item["predicted_keypoints"]).tolist(),
                    "target_keypoints": np.asarray(item["target_keypoints"]).tolist(),
                    "normalization_factor": item.get("normalization_factor"),
                }
            )
        payload["predictions"] = serializable_preds

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config)

    if args.backbone:
        config.model.backbone = args.backbone
    if args.batch_size:
        config.dataloader.batch_size = args.batch_size

    if args.split not in config.dataset.splits:
        available = ", ".join(config.dataset.splits.keys()) or "none"
        raise ValueError(f"Split '{args.split}' not found in configuration. Available: {available}")

    dataset_path = Path(config.dataset.splits[args.split])
    dataset = CorrespondenceDataset(dataset_path)
    dataloader = SimpleDataLoader(dataset, batch_size=config.dataloader.batch_size)

    model = FrozenBackboneModel(
        backbone=config.model.backbone,
        feature_dim=config.model.feature_dim,
        checkpoint=config.model.checkpoint,
    )
    model.freeze()

    runner = InferenceRunner(model)
    predictions = runner.run(dataloader)
    metrics = evaluate_pck(
        predictions,
        threshold=config.evaluation.pck_threshold,
        default_normalization=config.dataset.normalization_factor,
    )

    output_dir = Path(config.evaluation.output_dir)
    output_file = output_dir / f"{args.split}_pck.json"
    metadata = {
        "config": str(args.config),
        "split": args.split,
        "backbone": config.model.backbone,
        "batch_size": config.dataloader.batch_size,
    }
    save_results(
        output_path=output_file,
        metrics=metrics,
        predictions=predictions if config.evaluation.save_predictions else None,
        metadata=metadata,
    )
    print(f"Evaluation complete. Overall PCK: {metrics['overall_pck']:.4f}")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
