"""Training script for fine-tuning semantic correspondence models."""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from dataset.spair import SPairDataset, compute_pck
from models.finetuner import FinetunableBackbone
from models.loss import CorrespondenceLoss


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model
    backbone_name: str = 'dinov2_vitb14'
    num_layers_to_finetune: int = 2
    
    # Data
    data_root: str = 'data/SPair-71k'
    batch_size: int = 8
    num_workers: int = 4
    
    # Training
    num_epochs: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    
    # Loss
    loss_type: str = 'cosine'
    negative_margin: float = 0.2
    
    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 50
    val_interval: int = 500
    save_interval: int = 1000
    
    # Paths
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    experiment_name: Optional[str] = None
    
    def __post_init__(self):
        if self.experiment_name is None:
            self.experiment_name = f"{self.backbone_name}_ft{self.num_layers_to_finetune}"


class Trainer:
    """Training orchestrator."""
    
    def __init__(self, config: TrainingConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Create output directories
        self.ckpt_dir = Path(config.checkpoint_dir) / config.experiment_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.ckpt_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        # Initialize model
        self.model = FinetunableBackbone(
            backbone_name=config.backbone_name,
            num_layers_to_finetune=config.num_layers_to_finetune,
            device=device
        )
        
        # Loss
        self.criterion = CorrespondenceLoss(
            loss_type=config.loss_type,
            negative_margin=config.negative_margin
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Data
        self.train_loader = self._create_dataloader('train')
        self.val_loader = self._create_dataloader('val')
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0
    
    def _create_dataloader(self, split: str) -> DataLoader:
        """Create dataloader for given split."""
        dataset = SPairDataset(
            root=self.config.data_root,
            split=split,
            size='large',
            long_side=518,
            normalize=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size if split == 'train' else 1,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        warmup_steps = self.config.warmup_epochs * len(self.train_loader)
        total_steps = self.config.num_epochs * len(self.train_loader)
        
        warmup = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
        
        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps
        )
        
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps]
        )
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            src_img = batch['src_img'].to(self.device)
            tgt_img = batch['tgt_img'].to(self.device)
            src_kps = batch['src_kps'].to(self.device)
            tgt_kps = batch['tgt_kps'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            # Forward
            src_features = self.model(src_img)
            tgt_features = self.model(tgt_img)
            
            # Loss
            loss = self.criterion(
                src_features, tgt_features,
                src_kps, tgt_kps, valid_mask,
                patch_size=self.model.stride
            )
            
            # Backward
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            n_batches += 1
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                avg_loss = epoch_loss / n_batches
                lr = self.optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
            
            # Validation
            if self.global_step % self.config.val_interval == 0:
                val_loss = self.validate()
                self.val_losses.append({'step': self.global_step, 'loss': val_loss})
                self.model.train()
            
            # Checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(epoch, batch_idx)
        
        return epoch_loss / n_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            src_img = batch['src_img'].to(self.device)
            tgt_img = batch['tgt_img'].to(self.device)
            src_kps = batch['src_kps'].to(self.device)
            tgt_kps = batch['tgt_kps'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            src_features = self.model(src_img)
            tgt_features = self.model(tgt_img)
            
            loss = self.criterion(
                src_features, tgt_features,
                src_kps, tgt_kps, valid_mask,
                patch_size=self.model.stride
            )
            
            val_loss += loss.item()
            n_batches += 1
        
        avg_val_loss = val_loss / n_batches
        print(f"\n Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, batch_idx: int):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        ckpt_path = self.ckpt_dir / f'checkpoint_step{self.global_step}.pt'
        torch.save(checkpoint, ckpt_path)
        print(f"\n Saved: {ckpt_path}")
    
    def train(self):
        """Full training loop."""
        print(f"\n Training: {self.config.experiment_name}")
        
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f"\n Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            
            val_loss = self.validate()
            self.save_checkpoint(epoch, len(self.train_loader))
        
        print(f"\n Training complete!")


def main() -> None:
    """Main training routine."""
    parser = argparse.ArgumentParser(description='Fine-tune semantic correspondence model')
    parser.add_argument('--backbone', type=str, default='dinov2_vitb14',
                        choices=['dinov2_vitb14', 'dinov3_vitb16'])
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of layers to finetune')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-root', type=str, default='data/SPair-71k')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        backbone_name=args.backbone,
        num_layers_to_finetune=args.layers,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(config, device=device)
    trainer.train()


if __name__ == "__main__":
    main()
