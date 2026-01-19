import os
import pandas as pd
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path


class PFWillowDataset(Dataset):
    """
    PF-WILLOW dataset for semantic correspondence.
    
    Args:
        root (str): Root directory of PF-WILLOW dataset
        long_side (int): Resize images so that max(H,W) = long_side
        normalize (bool): Apply ImageNet normalization
    """
    
    N_KEYPOINTS = 10  # Fixed number of keypoints
    
    def __init__(self, root, long_side=518, normalize=True):
        super().__init__()
        self.root = Path(root)
        self.long_side = long_side
        self.normalize = normalize
        
        # Check dataset exists
        assert self.root.exists(), f"Dataset not found at {root}"
        
        # Load test pairs
        pairs_file = self.root / "test_pairs.csv"
        assert pairs_file.exists(), f"test_pairs.csv not found at {pairs_file}"
        
        self.pairs = pd.read_csv(pairs_file)
        
        # Extract unique categories from folder names
        self.categories = self._get_categories()
        
        print(f"Loaded {len(self.pairs)} test pairs from PF-WILLOW")
        print(f"Found categories: {self.categories}")
        
        # Setup transforms
        self.img_transforms = self._build_transforms()
    
    def _get_categories(self):
        """Extract category names from folder structure"""
        folders = [d.name for d in self.root.iterdir() if d.is_dir()]
        # Remove suffixes like (woC), (wC), (S), (M), (G)
        categories = set()
        for folder in folders:
            # Extract base category name (before parenthesis)
            if '(' in folder:
                cat = folder.split('(')[0]
            else:
                cat = folder
            categories.add(cat)
        
        return sorted(list(categories))
    
    def _build_transforms(self):
        """Build image transformation pipeline"""
        transforms = [T.ToTensor()]
        
        if self.normalize:
            # ImageNet normalization
            transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
            )
        
        return T.Compose(transforms)
    
    def __len__(self):
        return len(self.pairs)
    
    def _resize_with_aspect_ratio(self, img, kps=None):
        """Resize image maintaining aspect ratio"""
        w, h = img.size
        
        # Calculate new size
        if h > w:
            new_h = self.long_side
            new_w = int(w * (new_h / h))
        else:
            new_w = self.long_side
            new_h = int(h * (new_w / w))
        
        # Resize image
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        # Scale keypoints if provided
        if kps is not None:
            scale_x = new_w / w
            scale_y = new_h / h
            kps_resized = kps.copy()
            kps_resized[:, 0] *= scale_x
            kps_resized[:, 1] *= scale_y
            return img_resized, kps_resized
        
        return img_resized
    
    def _load_mat_annotation(self, mat_path):
        """Load keypoints from .mat file"""
        mat_data = sio.loadmat(str(mat_path))
        
        # Extract keypoints (10, 3) - [x, y, visibility]
        kps = mat_data['kps']  # Shape: (10, 3)
        
        # Separate coordinates and visibility
        coords = kps[:, :2]  # (10, 2)
        visibility = kps[:, 2]  # (10,)
        
        return coords, visibility
    
    def __getitem__(self, idx):
        """
        Returns:
            dict: {
                'src_img': (3, H, W),
                'tgt_img': (3, H, W),
                'src_kps': (N, 2),
                'tgt_kps': (N, 2),
                'valid_mask': (N,),
                'category': str,
                'src_size': (H, W),
                'tgt_size': (H, W)
            }
        """
        row = self.pairs.iloc[idx]
        
        # Parse pair info
        src_name = row['source_image']
        tgt_name = row['target_image']
        category = row['category']
        
        # Load images
        src_img = Image.open(self.root / 'JPEGImages' / src_name).convert('RGB')
        tgt_img = Image.open(self.root / 'JPEGImages' / tgt_name).convert('RGB')
        
        # Load annotations
        src_ann = self.root / 'Annotations' / src_name.replace('.jpg', '.mat')
        tgt_ann = self.root / 'Annotations' / tgt_name.replace('.jpg', '.mat')
        
        src_kps, src_vis = self._load_mat_annotation(src_ann)
        tgt_kps, tgt_vis = self._load_mat_annotation(tgt_ann)
        
        # Store original sizes
        src_size = torch.tensor(src_img.size[::-1])  # (H, W)
        tgt_size = torch.tensor(tgt_img.size[::-1])
        
        # Resize images and keypoints
        src_img, src_kps = self._resize_with_aspect_ratio(src_img, src_kps)
        tgt_img, tgt_kps = self._resize_with_aspect_ratio(tgt_img, tgt_kps)
        
        # Apply transforms
        src_img = self.img_transforms(src_img)
        tgt_img = self.img_transforms(tgt_img)
        
        # Create valid mask (both keypoints must be visible)
        valid_mask = (src_vis > 0) & (tgt_vis > 0)
        
        return {
            'src_img': src_img,
            'tgt_img': tgt_img,
            'src_kps': torch.from_numpy(src_kps).float(),
            'tgt_kps': torch.from_numpy(tgt_kps).float(),
            'valid_mask': torch.from_numpy(valid_mask).bool(),
            'category': category,
            'src_size': src_size,
            'tgt_size': tgt_size,
            'pair_idx': idx
        }
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        # Stack images
        src_imgs = torch.stack([item['src_img'] for item in batch])
        tgt_imgs = torch.stack([item['tgt_img'] for item in batch])
        
        # Stack keypoints and masks
        src_kps = torch.stack([item['src_kps'] for item in batch])
        tgt_kps = torch.stack([item['tgt_kps'] for item in batch])
        valid_masks = torch.stack([item['valid_mask'] for item in batch])
        
        # Keep as lists
        categories = [item['category'] for item in batch]
        
        return {
            'src_img': src_imgs,
            'tgt_img': tgt_imgs,
            'src_kps': src_kps,
            'tgt_kps': tgt_kps,
            'valid_mask': valid_masks,
            'category': categories,
            'src_size': torch.stack([item['src_size'] for item in batch]),
            'tgt_size': torch.stack([item['tgt_size'] for item in batch])
        }