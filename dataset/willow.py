import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.io as sio
from torchvision import transforms


class PFWillowDataset(Dataset):
    """PF-WILLOW dataset for semantic correspondence."""
    
    N_KEYPOINTS = 10
    
    def __init__(self, root, long_side=518, normalize=True):
        super().__init__()
        self.root = Path(root)
        self.long_side = long_side
        self.normalize = normalize
        
        assert self.root.exists(), f"Dataset not found at {root}"
        
        pairs_file = self.root / "test_pairs.csv"
        assert pairs_file.exists(), f"test_pairs.csv not found"
        
        self.pairs = pd.read_csv(pairs_file)
        
        # Detect column names
        source_col = self.pairs.columns[0]
        target_col = self.pairs.columns[1]
        self.source_col = source_col
        self.target_col = target_col
        
        # Extract BOTH category and subset (keep parentheses info)
        def extract_info(path_str):
            """Extract both category and full subset name"""
            try:
                parts = str(path_str).split('/')
                if len(parts) >= 2:
                    subset_full = parts[1]  # "car(G)"
                    category = subset_full.split('(')[0]  # "car"
                    return pd.Series({'category': category, 'subset': subset_full})
                return pd.Series({'category': 'unknown', 'subset': 'unknown'})
            except:
                return pd.Series({'category': 'unknown', 'subset': 'unknown'})
        
        print("Extracting category and subset from image paths...")
            
                        
        info = self.pairs[source_col].apply(extract_info)
        self.pairs['category'] = info['category']
        self.pairs['subset'] = info['subset']
            
        print(f"Categories: {sorted(self.pairs['category'].unique().tolist())}")
        print(f"Subsets: {sorted(self.pairs['subset'].unique().tolist())}")
        
        self.categories = sorted(self.pairs['category'].unique().tolist())
        self.subsets = sorted(self.pairs['subset'].unique().tolist())
        
        print(f"\nLoaded {len(self.pairs)} pairs")
        print(f"  {len(self.categories)} categories: {self.categories}")
        print(f"  {len(self.subsets)} subsets")
        
        self.img_transforms = self._build_transforms()
    
    def _build_transforms(self):
        transforms_list = [transforms.ToTensor()]
        if self.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        return transforms.Compose(transforms_list)
    
    def __len__(self):
        return len(self.pairs)
    
    def _resize_with_aspect_ratio(self, img, kps=None):
        W, H = img.size
        max_dim = max(H, W)
        scale = self.long_side / max_dim
        new_H = int(H * scale)
        new_W = int(W * scale)
        resized_img = img.resize((new_W, new_H), Image.BILINEAR)
        
        if kps is not None:
            resized_kps = kps * scale
            return resized_img, resized_kps, (scale, scale)
        return resized_img, (scale, scale)
    
    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        
        img_a_path = row[self.source_col]
        img_b_path = row[self.target_col]
        
        img_a_parts = img_a_path.split('/')
        subset_a = img_a_parts[1]
        filename_a = img_a_parts[2]
        
        img_b_parts = img_b_path.split('/')
        subset_b = img_b_parts[1]
        filename_b = img_b_parts[2]
        
        src_path = self.root / subset_a / filename_a
        tgt_path = self.root / subset_b / filename_b
        
        src_img = Image.open(src_path).convert('RGB')
        tgt_img = Image.open(tgt_path).convert('RGB')
        src_size_orig = src_img.size
        tgt_size_orig = tgt_img.size
        
        try:
            src_kps_x = np.array([row[f'XA{i}'] for i in range(1, 11)])
            src_kps_y = np.array([row[f'YA{i}'] for i in range(1, 11)])
        except KeyError:
            src_kps_x = row.iloc[2:12].values.astype(float)
            src_kps_y = row.iloc[12:22].values.astype(float)
        
        src_kps = np.stack([src_kps_x, src_kps_y], axis=1).astype(np.float32)
        
        tgt_mat_path = tgt_path.with_suffix('.mat')
        
        if tgt_mat_path.exists():
            mat_data = sio.loadmat(str(tgt_mat_path))
            
            if 'kps' in mat_data:
                tgt_kps = mat_data['kps']
            elif 'pts' in mat_data:
                tgt_kps = mat_data['pts']
            else:
                tgt_kps = None
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        arr = mat_data[key]
                        if hasattr(arr, 'shape') and arr.shape in [(2, 10), (10, 2)]:
                            tgt_kps = arr
                            break
                if tgt_kps is None:
                    tgt_kps = src_kps.copy()
            
            if tgt_kps.shape == (2, 10):
                tgt_kps = tgt_kps.T
            tgt_kps = tgt_kps.astype(np.float32)
        else:
            tgt_kps = src_kps.copy()
        
        valid_src = (
            (src_kps[:, 0] >= 0) & (src_kps[:, 0] < src_size_orig[0]) &
            (src_kps[:, 1] >= 0) & (src_kps[:, 1] < src_size_orig[1])
        )
        valid_tgt = (
            (tgt_kps[:, 0] >= 0) & (tgt_kps[:, 0] < tgt_size_orig[0]) &
            (tgt_kps[:, 1] >= 0) & (tgt_kps[:, 1] < tgt_size_orig[1])
        )
        valid_mask = valid_src & valid_tgt
        
        src_img_resized, src_kps_resized, _ = self._resize_with_aspect_ratio(
            src_img, torch.from_numpy(src_kps)
        )
        tgt_img_resized, tgt_kps_resized, _ = self._resize_with_aspect_ratio(
            tgt_img, torch.from_numpy(tgt_kps)
        )
        
        src_img_tensor = self.img_transforms(src_img_resized)
        tgt_img_tensor = self.img_transforms(tgt_img_resized)
        
        return {
            'src_img': src_img_tensor,
            'tgt_img': tgt_img_tensor,
            'src_kps': src_kps_resized,
            'tgt_kps': tgt_kps_resized,
            'valid_mask': torch.from_numpy(valid_mask),
            'category': row['category'],      # "car"
            'subset': row['subset'],          # "car(G)" ← Include full subset
            'src_size': torch.tensor(src_img_resized.size[::-1]),
            'tgt_size': torch.tensor(tgt_img_resized.size[::-1])
        }
    
    @staticmethod
    def collate_fn(batch):
        return {
            'src_img': torch.stack([b['src_img'] for b in batch]),
            'tgt_img': torch.stack([b['tgt_img'] for b in batch]),
            'src_kps': torch.stack([b['src_kps'] for b in batch]),
            'tgt_kps': torch.stack([b['tgt_kps'] for b in batch]),
            'valid_mask': torch.stack([b['valid_mask'] for b in batch]),
            'category': [b['category'] for b in batch],
            'subset': [b['subset'] for b in batch],  # ← Include subset
            'src_size': torch.stack([b['src_size'] for b in batch]),
            'tgt_size': torch.stack([b['tgt_size'] for b in batch])
        }