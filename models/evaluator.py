import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
import time

from utils import compute_pck


class UnifiedEvaluator:
    """Evaluation engine for semantic correspondence backbones."""

    def __init__(
        self,
        dataloader,
        device: str = 'cuda',
        thresholds: List[float] = None
    ):
        """Initialize evaluator.
        
        Args:
            dataloader: PyTorch DataLoader with test data
            device: Device to use ('cuda' or 'cpu')
            thresholds: PCK thresholds (default: [0.05, 0.10, 0.15, 0.20])
        """
        self.dataloader = dataloader
        self.device = device
        self.thresholds = thresholds or [0.05, 0.10, 0.15, 0.20]
        self.results = {}

    def evaluate(
        self,
        matcher,
        backbone_name: str,
        num_samples: int = None,
        show_progress: bool = True
    ) -> Dict:
        """Evaluate a matcher on the dataset.
        
        Args:
            matcher: CorrespondenceMatcher instance
            backbone_name: Display name for results
            num_samples: Max samples to evaluate (None = all)
            show_progress: Show progress bar
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATING: {backbone_name}")
        print('='*70)

        # Storage
        per_keypoint = defaultdict(list)
        per_cat_keypoint = defaultdict(lambda: defaultdict(list))
        per_cat_image = defaultdict(lambda: defaultdict(list))
        per_image = defaultdict(list)
        inference_times = []
        n_processed = 0

        # Set backbone to eval mode
        was_training = matcher.backbone.training
        matcher.backbone.eval()

        nks_total = 0
        # Evaluation loop
        pbar = tqdm(self.dataloader, desc=backbone_name) if show_progress else self.dataloader
        for batch in pbar:
            if num_samples and n_processed >= num_samples:
                break

            src_img = batch['src_img'].to(self.device)  # (1, 3, H_resized, W_resized)
            tgt_img = batch['tgt_img'].to(self.device)  # (1, 3, H_resized, W_resized)
            
            tgt_orig_size = batch['tgt_orig_size'][0]  # (H_orig, W_orig) tensor
            tgt_scale = batch['tgt_scale'][0].item()    # float scalar
            
            # Extract keypoints and masks
            src_kps = batch['src_kps'][0]  # (N, 2) in resized coords
            tgt_kps = batch['tgt_kps'][0]  # (N, 2) in resized coords
            valid_mask = batch['valid_mask'][0]
            category = batch['category'][0]

            # Filter valid keypoints
            src_kps_valid = src_kps[valid_mask]
            tgt_kps_valid = tgt_kps[valid_mask]

            if len(src_kps_valid) == 0:
                continue

            # Predict with timing
            start = time.time()
            tgt_kps_pred = matcher.match(src_img, tgt_img, src_kps_valid)
            inference_times.append(time.time() - start)

            H_orig, W_orig = tgt_orig_size.tolist()
            
            bbox_size = None
            if 'tgt_bbox' in batch:
                tgt_bbox = batch['tgt_bbox'][0]  # Bbox in resized coords
                tgt_bbox_orig = tgt_bbox / tgt_scale  # Convert to original coords
                bbox_h = (tgt_bbox_orig[3] - tgt_bbox_orig[1]).item()
                bbox_w = (tgt_bbox_orig[2] - tgt_bbox_orig[0]).item()
                if bbox_h > 0 and bbox_w > 0:
                    bbox_size = (bbox_h, bbox_w)
            
            tgt_kps_pred_orig = (tgt_kps_pred / tgt_scale).detach().cpu()
            tgt_kps_gt_orig   = (tgt_kps_valid / tgt_scale).detach().cpu()
            pck_scores = compute_pck(
                tgt_kps_pred_orig, 
                tgt_kps_gt_orig, 
                bbox_size=bbox_size,              #  Bbox in original coords (if available)
                image_size=(H_orig, W_orig),      #  Use original size
                thresholds=self.thresholds
            )

            # Store results
            for metric, value in pck_scores.items():
                per_keypoint[metric].extend(value[0].cpu().tolist())
                per_cat_keypoint[category][metric].extend(value[0].cpu().tolist())
                per_cat_image[category][metric].append(value[1])
                per_image[metric].append(value[1])
            nks_total += len(tgt_kps_valid)
            n_processed += 1

            # Update progress
            if show_progress and len(per_keypoint['PCK@0.10']) > 0:
                avg_pck = np.mean(per_keypoint['PCK@0.10'])
                pbar.set_postfix({'PCK@0.10': f'{avg_pck:.4f}'})

        # Aggregate results
        results = self._aggregate_results(
            backbone_name, per_keypoint, per_cat_keypoint, per_cat_image, per_image, inference_times, n_processed, nks_total
        )

        self.results[backbone_name] = results
        self._print_summary(results)
        
        # Restore backbone training state
        if was_training:
            matcher.backbone.train()
        
        # Cleanup
        torch.cuda.empty_cache()

        return results

    def _aggregate_results(self, name, per_keypoint, per_cat_keypoint, per_cat_image, per_image, times, n_pairs, nks_total):
        """Aggregate metrics into structured results."""
        
        results = {
            'name': name,
            'num_pairs': n_pairs,
            'inference_time_ms': np.mean(times) * 1000 if times else 0,
            'overall_keypoint': {},
            'per_category_keypoint': {},
            'overall_image': {},
            'per_category_image': {}
        }

        # Overall metrics per keypoint
        for metric in [f'PCK@{t:.2f}' for t in self.thresholds]:
            values = np.array(per_keypoint[metric])  # Array of 0/1 for all keypoints
            results['overall_keypoint'][metric] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
            }

        # Per-category metrics per keypoint
        for cat, metrics in per_cat_keypoint.items():
            results['per_category_keypoint'][cat] = {}
            for metric in [f'PCK@{t:.2f}' for t in self.thresholds]:
                per_cat_values = np.array(metrics[metric])
                results['per_category_keypoint'][cat][metric] = per_cat_values.mean().item()

        # Overall metrics per image
        for metric in [f'PCK@{t:.2f}' for t in self.thresholds]:
            values = np.array(per_image[metric])  # Array of 0/1 for all images
            results['overall_image'][metric] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
            }
        # Per-category metrics per image
        for cat, metrics in per_cat_image.items():
            results['per_category_image'][cat] = {}
            for metric in [f'PCK@{t:.2f}' for t in self.thresholds]:
                per_cat_values = np.array(metrics[metric])
                results['per_category_image'][cat][metric] = per_cat_values.mean().item()
                
        return results

    def _print_summary(self, results):
        """Print evaluation summary."""
        
        print(f"\n{results['name']} Results:")
        print("-" * 70)

        for metric, values in results['overall_keypoint'].items():
            mean_val = values['mean']
            std_val = values['std']
            print(f"   {metric}: {mean_val:.4f} ± {std_val:.4f} ({mean_val*100:.2f}%)")
        
        for metric, values in results['overall_image'].items():
            mean_val = values['mean']
            std_val = values['std']
            print(f"   [Image] {metric}: {mean_val:.4f} ± {std_val:.4f} ({mean_val*100:.2f}%)")

        print(f"\n   Avg inference time: {results['inference_time_ms']:.2f} ms/pair")
        print(f"    Evaluated on {results['num_pairs']} pairs")

