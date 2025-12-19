"""Evaluation script for semantic correspondence baseline."""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.spair import SPairDataset
from src.features.dino import get_feature_extractor
from src.matching.matcher import Matcher
from src.metrics.pck import PCKMetric
from utils.seed import set_seed
from utils.logger import Logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate(config: dict):
    """Run evaluation.
    
    Args:
        config: Configuration dictionary
    """
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize logger
    logger = Logger(
        project=config['logging']['project'],
        config=config,
        use_wandb=config['logging']['use_wandb'],
        run_name=config['logging']['run_name']
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = SPairDataset(
        root_dir=config['dataset']['root_dir'],
        split=config['dataset']['split'],
        category=config['dataset']['category'],
        img_size=config['dataset']['img_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers']
    )
    
    print(f"Loaded {len(dataset)} image pairs")
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = get_feature_extractor(
        model_type=config['features']['model_type'],
        model_name=config['features']['model_name'],
        layer=config['features']['layer'],
        facet=config['features']['facet'],
        use_cls=config['features']['use_cls']
    ).to(device)
    
    # Initialize matcher
    matcher = Matcher(
        matching_type=config['matching']['type'],
        use_ransac=config['matching']['use_ransac'],
        ransac_threshold=config['matching']['ransac_threshold']
    )
    
    # Initialize metrics
    pck_metric = PCKMetric(alphas=config['evaluation']['alphas'])
    
    # Evaluation loop
    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            src_img = batch['src_img'].to(device)
            trg_img = batch['trg_img'].to(device)
            src_kps = batch['src_kps'].squeeze(0)  # Remove batch dimension
            trg_kps = batch['trg_kps'].squeeze(0)
            
            # Skip if no keypoints
            if len(src_kps) == 0 or len(trg_kps) == 0:
                continue
            
            # Extract features
            src_desc = feature_extractor.extract_descriptors(src_img, return_spatial=True)
            trg_desc = feature_extractor.extract_descriptors(trg_img, return_spatial=True)
            
            # Remove batch dimension
            src_desc = src_desc.squeeze(0)  # [D, H, W]
            trg_desc = trg_desc.squeeze(0)  # [D, H, W]
            
            # Match descriptors
            matched_src_kps, matched_trg_kps = matcher.match(
                src_desc, trg_desc, src_kps
            )
            
            # Compute PCK
            if len(matched_trg_kps) > 0:
                # Get image size
                img_size = (config['dataset']['img_size'], config['dataset']['img_size'])
                
                pck_metric.update(matched_trg_kps, trg_kps, img_size)
            
            # Log intermediate results every 10 batches
            if (batch_idx + 1) % 10 == 0:
                intermediate_results = pck_metric.compute()
                logger.log(intermediate_results, step=batch_idx + 1)
    
    # Compute final metrics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(pck_metric.get_summary())
    
    # Log final results
    final_results = pck_metric.compute()
    logger.log(final_results)
    
    # Finish logging
    logger.finish()
    
    return final_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate semantic correspondence baseline")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Override dataset root directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable WandB logging'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config['dataset']['root_dir'] = args.data_dir
    
    if args.device is not None:
        config['device'] = args.device
    
    if args.no_wandb:
        config['logging']['use_wandb'] = False
    
    # Run evaluation
    evaluate(config)


if __name__ == '__main__':
    main()
