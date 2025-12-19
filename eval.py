import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


ImageSize = Tuple[int, int]  # (width, height)
Keypoints = List[Tuple[float, float]]


def _load_json(path: Path) -> List[Dict]:
    with path.open("r") as f:
        return json.load(f)


def _save_json(path: Path, content: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(list(content), f, indent=2)


def _scale_keypoints(
    keypoints: Keypoints,
    original_size: ImageSize,
    resized_size: ImageSize,
) -> Keypoints:
    if not keypoints:
        return []

    src_w, src_h = original_size
    dst_w, dst_h = resized_size
    scale_x = (dst_w - 1) / max(src_w - 1, 1)
    scale_y = (dst_h - 1) / max(src_h - 1, 1)

    return [(x * scale_x, y * scale_y) for x, y in keypoints]


def _keypoints_to_grid(
    keypoints: Keypoints, image_size: ImageSize, device: torch.device
) -> torch.Tensor:
    if not keypoints:
        return torch.empty(1, 0, 1, 2, device=device)

    img_w, img_h = image_size
    # Normalize to [-1, 1] for grid_sample.
    norm_x = torch.tensor(
        [kp[0] * 2 / max(img_w - 1, 1) - 1 for kp in keypoints],
        device=device,
        dtype=torch.float32,
    )
    norm_y = torch.tensor(
        [kp[1] * 2 / max(img_h - 1, 1) - 1 for kp in keypoints],
        device=device,
        dtype=torch.float32,
    )
    grid = torch.stack([norm_x, norm_y], dim=-1)
    return grid.view(1, -1, 1, 2)


def _feature_at_keypoints(
    feature_map: torch.Tensor,
    keypoints: Keypoints,
    image_size: ImageSize,
    device: torch.device,
) -> torch.Tensor:
    if not keypoints:
        return torch.empty(0, feature_map.shape[1], device=device)

    grid = _keypoints_to_grid(keypoints, image_size, device)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        align_corners=True,
    )  # (1, C, N, 1)
    return sampled.squeeze(0).permute(1, 0, 2).reshape(len(keypoints), -1)


def _feature_argmax_to_image_coords(
    argmax_idx: torch.Tensor,
    feature_size: Tuple[int, int],
    image_size: ImageSize,
) -> Tuple[float, float]:
    feat_h, feat_w = feature_size
    y_idx = torch.div(argmax_idx, feat_w, rounding_mode="floor")
    x_idx = argmax_idx % feat_w

    img_w, img_h = image_size
    x_img = x_idx.float() / max(feat_w - 1, 1) * max(img_w - 1, 1)
    y_img = y_idx.float() / max(feat_h - 1, 1) * max(img_h - 1, 1)
    return float(x_img.item()), float(y_img.item())


@dataclass
class BaselineConfig:
    pairs_json: Path
    output_path: Path
    device: torch.device
    image_size: int = 320
    normalize_mean: Sequence[float] = (0.485, 0.456, 0.406)
    normalize_std: Sequence[float] = (0.229, 0.224, 0.225)


class FeatureExtractor(torch.nn.Module):
    """
    Minimal ResNet feature extractor returning the final convolutional block
    activations. The resulting feature maps are L2-normalized channel-wise.
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(backbone.children())[:-2]  # Remove avgpool + classifier.
        self.encoder = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return F.normalize(feats, p=2, dim=1)


class BaselinePredictor:
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.device = config.device
        self.feature_extractor = FeatureExtractor().to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(config.normalize_mean, config.normalize_std),
            ]
        )

    def _load_image(self, path: Path) -> Tuple[torch.Tensor, ImageSize, ImageSize]:
        image = Image.open(path).convert("RGB")
        original_size = image.size  # (width, height)
        image = self.transform(image)
        resized_size = (image.shape[2], image.shape[1])  # (width, height)
        return image, original_size, resized_size

    @torch.inference_mode()
    def predict_pair(self, sample: Dict) -> Dict:
        image_id = sample.get("id") or sample.get("image_id")
        src_img_path = Path(sample["source_image"])
        tgt_img_path = Path(sample["target_image"])
        keypoints = [(float(x), float(y)) for x, y in sample.get("source_keypoints", [])]
        valid_mask = sample.get("valid_mask")
        if valid_mask is None:
            valid_mask = [True] * len(keypoints)
        if len(valid_mask) < len(keypoints):
            valid_mask = valid_mask + [True] * (len(keypoints) - len(valid_mask))
        elif len(valid_mask) > len(keypoints):
            valid_mask = valid_mask[: len(keypoints)]

        src_img, src_orig_size, src_resized = self._load_image(src_img_path)
        tgt_img, tgt_orig_size, tgt_resized = self._load_image(tgt_img_path)

        src_keypoints = _scale_keypoints(keypoints, src_orig_size, src_resized)

        src_batch = src_img.unsqueeze(0).to(self.device)
        tgt_batch = tgt_img.unsqueeze(0).to(self.device)

        src_features = self.feature_extractor(src_batch)
        tgt_features = self.feature_extractor(tgt_batch)

        src_vectors = _feature_at_keypoints(
            src_features, src_keypoints, src_resized, self.device
        )
        tgt_flat = tgt_features.view(1, tgt_features.shape[1], -1).squeeze(0)

        predictions: List[Optional[Tuple[float, float]]] = []
        feat_h, feat_w = tgt_features.shape[-2:]
        for vec, is_valid in zip(src_vectors, valid_mask):
            if not is_valid:
                predictions.append(None)
                continue
            similarities = torch.einsum("nc,c->n", tgt_flat.transpose(0, 1), vec)
            argmax_idx = torch.argmax(similarities)
            pred_xy = _feature_argmax_to_image_coords(
                argmax_idx, (feat_h, feat_w), tgt_resized
            )
            # Map back to the original target image resolution.
            x_orig = pred_xy[0] / max(tgt_resized[0] - 1, 1) * max(tgt_orig_size[0] - 1, 1)
            y_orig = pred_xy[1] / max(tgt_resized[1] - 1, 1) * max(tgt_orig_size[1] - 1, 1)
            predictions.append((x_orig, y_orig))

        return {
            "image_id": image_id,
            "pred_keypoints": predictions,
            "valid_mask": valid_mask,
        }

    def run(self) -> None:
        pairs = _load_json(self.config.pairs_json)
        results = (self.predict_pair(sample) for sample in pairs)
        _save_json(self.config.output_path, results)


def parse_args() -> BaselineConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline semantic correspondence inference. The input JSON must be a "
            "list of dictionaries with keys: 'id' (or 'image_id'), 'source_image', "
            "'target_image', 'source_keypoints', and optional 'valid_mask'."
        )
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/pairs.json"),
        help="Path to the JSON file describing source/target pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/baseline_predictions.json"),
        help="Where to store the predicted keypoints.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
        help="Images are resized to a square of this size before feature extraction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device to use.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    return BaselineConfig(
        pairs_json=args.pairs,
        output_path=args.output,
        device=device,
        image_size=args.image_size,
    )


def main():
    config = parse_args()
    predictor = BaselinePredictor(config)
    predictor.run()


if __name__ == "__main__":
    main()
