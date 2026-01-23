import torch
from torch.utils.data import DataLoader
from dataset.willow import PFWillowDataset
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json


def compute_pck(pred_kps, gt_kps, img_size, thresholds=[0.05, 0.10, 0.15, 0.20]):
    """
    Compute PCK metric for semantic correspondence.
    
    Args:
        pred_kps: (N, 2) predicted keypoints
        gt_kps: (N, 2) ground truth keypoints
        img_size: (H, W) image size
        thresholds: list of alpha values [0.05, 0.10, 0.15, 0.20]
    
    Returns:
        dict: PCK scores for each threshold
    """
    # Compute distances
    distances = torch.norm(pred_kps - gt_kps, dim=-1)  # (N,)
    
    # Normalization factor: max(H, W)
    norm_factor = max(img_size[0], img_size[1])
    
    # Compute PCK for each threshold
    pck_dict = {}
    for alpha in thresholds:
        threshold = alpha * norm_factor
        correct = (distances <= threshold).float()
        pck = correct.mean().item() * 100
        pck_dict[f'PCK@{alpha:.2f}'] = pck
    
    return pck_dict


@torch.no_grad()
def evaluate_model(model, dataloader, device='cuda', thresholds=[0.05, 0.10, 0.15, 0.20]):
    """
    Evaluate trained model on PF-WILLOW dataset.
    
    Args:
        model: Your trained model from Task 2
        dataloader: PF-WILLOW DataLoader
        device: 'cuda' or 'cpu'
        thresholds: PCK thresholds to evaluate
    
    Returns:
        dict: Overall and per-category results
    """
    model.eval()
    model.to(device)
    
    # Initialize storage for all thresholds
    all_results = {f'PCK@{t:.2f}': [] for t in thresholds}
    category_results = {}
    
    print(f"\n{'='*70}")
    print("Evaluating on PF-WILLOW Dataset")
    print(f"{'='*70}")
    print(f"Thresholds: {thresholds}")
    print(f"Device: {device}\n")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        src_img = batch['src_img'].to(device)
        tgt_img = batch['tgt_img'].to(device)
        src_kps = batch['src_kps'].to(device)
        tgt_kps = batch['tgt_kps'].to(device)
        valid_mask = batch['valid_mask'].to(device)
        categories = batch['category']
        tgt_size = batch['tgt_size']
        
        # Predict correspondences using your trained model
        # Il model deve avere un metodo predict() o forward() che ritorna keypoints predetti
        pred_kps = model.predict(src_img, tgt_img, src_kps)
        
        # Evaluate each sample in batch
        batch_size = src_img.size(0)
        for i in range(batch_size):
            # Get valid keypoints
            valid_idx = valid_mask[i]
            
            if valid_idx.sum() == 0:
                continue
            
            pred_valid = pred_kps[i][valid_idx]
            gt_valid = tgt_kps[i][valid_idx]
            
            # Compute PCK for all thresholds
            pck = compute_pck(
                pred_valid.cpu(),
                gt_valid.cpu(),
                tgt_size[i].tolist(),
                thresholds=thresholds
            )
            
            # Accumulate results
            for key, value in pck.items():
                all_results[key].append(value)
            
            # Per-category results
            cat = categories[i]
            if cat not in category_results:
                category_results[cat] = {k: [] for k in pck.keys()}
            
            for key, value in pck.items():
                category_results[cat][key].append(value)
    
    # Compute averages
    final_results = {
        'overall': {k: np.mean(v) for k, v in all_results.items()},
        'per_category': {}
    }
    
    for cat, metrics in category_results.items():
        final_results['per_category'][cat] = {
            k: np.mean(v) for k, v in metrics.items()
        }
    
    return final_results


def print_results(results):
    """Pretty print evaluation results"""
    print("\n" + "="*70)
    print("PF-WILLOW EVALUATION RESULTS")
    print("="*70)
    
    # Overall results
    print("\n📊 Overall Performance:")
    print("-" * 70)
    for metric in ['PCK@0.05', 'PCK@0.10', 'PCK@0.15', 'PCK@0.20']:
        if metric in results['overall']:
            value = results['overall'][metric]
            print(f"  {metric}: {value:.2f}%")
    
    # Per-category results
    print("\n📋 Per-Category Performance:")
    print("-" * 70)
    for cat in sorted(results['per_category'].keys()):
        metrics = results['per_category'][cat]
        print(f"\n  {cat.upper()}:")
        for metric in ['PCK@0.05', 'PCK@0.10', 'PCK@0.15', 'PCK@0.20']:
            if metric in metrics:
                value = metrics[metric]
                print(f"    {metric}: {value:.2f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "checkpoints/best_model_task2.ckpt"  # Your trained model from Task 2
    DATA_ROOT = "data/PF-WILLOW"
    BATCH_SIZE = 1  # Use batch_size=1 for simplicity
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("🎯 Loading PF-WILLOW Dataset...")
    
    # Load dataset
    dataset = PFWillowDataset(
        root=DATA_ROOT,
        long_side=518,
        normalize=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=PFWillowDataset.collate_fn
    )
    
    print(f"✅ Loaded {len(dataset)} test pairs")
    
    # Load your trained model from Task 2
    print(f"\n🔧 Loading trained model from: {CHECKPOINT_PATH}")
    
    # IMPORTANTE: Sostituisci con il tuo import del modello Task 2
    # Esempio:
    # from models.correspondence_model import CorrespondenceModel
    # model = CorrespondenceModel.load_from_checkpoint(CHECKPOINT_PATH)
    
    # Oppure se usi PyTorch Lightning:
    import pytorch_lightning as pl
    from your_model_file import YourModelClass  # <- AGGIORNA QUESTO
    
    model = YourModelClass.load_from_checkpoint(CHECKPOINT_PATH)
    
    print(f"✅ Model loaded successfully\n")
    
    # Evaluate with 4 PCK thresholds
    results = evaluate_model(
        model, 
        dataloader, 
        device=DEVICE,
        thresholds=[0.05, 0.10, 0.15, 0.20]
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path("results") / "pf_willow_task4_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")