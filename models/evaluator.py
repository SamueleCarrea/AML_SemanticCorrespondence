"""Unified evaluation engine."""

import torch
import numpy as np
import pandas as pd
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
        all_pck = defaultdict(list)
        per_category = defaultdict(lambda: defaultdict(list))
        inference_times = []
        n_processed = 0

        # Evaluation loop
        pbar = tqdm(self.dataloader, desc=backbone_name) if show_progress else self.dataloader

        for batch in pbar:
            if num_samples and n_processed >= num_samples:
                break

            # Extract data
            src_img = batch['src_img'].to(self.device)
            tgt_img = batch['tgt_img'].to(self.device)
            src_kps = batch['src_kps'][0]  # (N, 2)
            tgt_kps = batch['tgt_kps'][0]
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

            # Compute metrics
            H, W = tgt_img.shape[2:]
            pck_scores = compute_pck(tgt_kps_pred, tgt_kps_valid, (H, W), self.thresholds)

            # Store results
            for metric, value in pck_scores.items():
                all_pck[metric].append(value)
                per_category[category][metric].append(value)

            n_processed += 1

            # Update progress
            if show_progress and len(all_pck['PCK@0.10']) > 0:
                avg_pck = np.mean(all_pck['PCK@0.10'])
                pbar.set_postfix({'PCK@0.10': f'{avg_pck:.4f}'})

        # Aggregate results
        results = self._aggregate_results(
            backbone_name, all_pck, per_category, inference_times, n_processed
        )

        self.results[backbone_name] = results
        self._print_summary(results)
        
        # Cleanup
        del matcher.backbone
        torch.cuda.empty_cache()

        return results

    def _aggregate_results(self, name, all_pck, per_category, times, n_pairs):
        """Aggregate metrics into structured results."""
        
        results = {
            'name': name,
            'num_pairs': n_pairs,
            'inference_time_ms': np.mean(times) * 1000 if times else 0,
            'overall': {},
            'per_category': {}
        }

        # Overall metrics
        for metric in [f'PCK@{t:.2f}' for t in self.thresholds]:
            values = all_pck[metric]
            results['overall'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
            }

        # Per-category metrics
        for cat, metrics in per_category.items():
            results['per_category'][cat] = {}
            for metric in [f'PCK@{t:.2f}' for t in self.thresholds]:
                results['per_category'][cat][metric] = np.mean(metrics[metric])

        return results

    def _print_summary(self, results):
        """Print evaluation summary."""
        
        print(f"\n{results['name']} Results:")
        print("-" * 70)

        for metric, values in results['overall'].items():
            mean_val = values['mean']
            std_val = values['std']
            print(f"   {metric}: {mean_val:.4f} ± {std_val:.4f} ({mean_val*100:.2f}%)")

        print(f"\n   ⏱ Avg inference time: {results['inference_time_ms']:.2f} ms/pair")
        print(f"   📊 Evaluated on {results['num_pairs']} pairs")

    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison DataFrame from all results."""
        
        if not self.results:
            print(" No evaluation results yet")
            return None

        rows = []
        for name, res in self.results.items():
            row = {
                'Backbone': res['name'],
                'Pairs': res['num_pairs'],
                'Time (ms)': f"{res['inference_time_ms']:.1f}",
            }

            for metric, vals in res['overall'].items():
                row[metric] = f"{vals['mean']:.4f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        print("\n" + "="*70)
        print("FINAL COMPARISON")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)

        return df
