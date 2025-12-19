"""Simple evaluation script with a progress-barred inference loop."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.device import choose_device, move_batch_to_device
from utils.logger import get_logger
from utils.seed import set_seed


def build_dummy_dataloader(batch_size: int = 8, num_batches: int = 10) -> DataLoader:
    """Create a dummy dataloader for demonstration purposes."""

    inputs = torch.randn(num_batches * batch_size, 4)
    dataset = TensorDataset(inputs)
    return DataLoader(dataset, batch_size=batch_size)


def run_inference(
    model: nn.Module, dataloader: Iterable[Tuple[torch.Tensor]], device: torch.device
) -> torch.Tensor:
    """Run inference with a progress bar and return concatenated outputs."""

    outputs = []
    model.eval()
    with torch.no_grad():
        for (batch,) in tqdm(dataloader, desc="Inference", unit="batch"):
            batch = move_batch_to_device(batch, device)
            preds = model(batch)
            outputs.append(preds.cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    logger = get_logger("evaluation")
    set_seed(42, deterministic=True)
    device = choose_device()
    logger.info(f"Using device: {device}")

    model = nn.Sequential(nn.Linear(4, 2), nn.Softmax(dim=-1)).to(device)
    dataloader = build_dummy_dataloader()

    logger.info("Starting inference…")
    outputs = run_inference(model, dataloader, device)
    logger.info("Inference completed")
    logger.info(f"Collected {len(outputs)} predictions")


if __name__ == "__main__":
    main()
