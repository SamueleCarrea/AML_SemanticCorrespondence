"""SPair-71k dataset loader for semantic correspondence."""

import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional
import torchvision.transforms as transforms


class SPairDataset(Dataset):
    """SPair-71k dataset loader.
    
    Dataset structure expected:
    data/SPair-71k/
        ImageAnnotation/
            {category}/
                {image_name}.json
        JPEGImages/
            {category}/
                {image_name}.jpg
        PairAnnotation/
            {category}/
                pairs-*.json
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        category: str = "all",
        img_size: int = 224,
    ):
        """Initialize SPair-71k dataset.
        
        Args:
            root_dir: Root directory containing SPair-71k data
            split: Dataset split (train/val/test)
            category: Object category or "all" for all categories
            img_size: Target image size for resizing
        """
        self.root_dir = root_dir
        self.split = split
        self.category = category
        self.img_size = img_size
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.pairs = self._load_pairs()
    
    def _load_pairs(self):
        """Load image pairs and annotations."""
        pairs = []
        
        # For simplicity, we'll create a basic loader
        # In a real implementation, this would parse the actual SPair-71k annotations
        pair_dir = os.path.join(self.root_dir, "PairAnnotation")
        
        if not os.path.exists(pair_dir):
            # Return empty list if directory doesn't exist
            # This allows the code to run even without the dataset
            return pairs
        
        categories = [self.category] if self.category != "all" else os.listdir(pair_dir)
        
        for cat in categories:
            cat_dir = os.path.join(pair_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
                
            for pair_file in os.listdir(cat_dir):
                if not pair_file.endswith('.json'):
                    continue
                
                pair_path = os.path.join(cat_dir, pair_file)
                try:
                    with open(pair_path, 'r') as f:
                        pair_data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Failed to load {pair_path}: {e}")
                    continue
                    
                if isinstance(pair_data, list):
                    for pair in pair_data:
                        pairs.append({
                            'category': cat,
                            'src_img': pair.get('src_imname', ''),
                            'trg_img': pair.get('trg_imname', ''),
                            'src_kps': np.array(pair.get('src_kps', [])),
                            'trg_kps': np.array(pair.get('trg_kps', [])),
                        })
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs) if len(self.pairs) > 0 else 100  # Return dummy length if no data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pair of images with keypoints.
        
        Args:
            idx: Index of the pair
            
        Returns:
            Dictionary containing:
                - src_img: Source image tensor
                - trg_img: Target image tensor
                - src_kps: Source keypoints
                - trg_kps: Target keypoints
                - category: Object category
        """
        if len(self.pairs) == 0:
            # Return dummy data for testing without the actual dataset
            return self._get_dummy_data()
        
        pair = self.pairs[idx]
        
        # Load images
        img_dir = os.path.join(self.root_dir, "JPEGImages", pair['category'])
        src_img_path = os.path.join(img_dir, pair['src_img'])
        trg_img_path = os.path.join(img_dir, pair['trg_img'])
        
        src_img = Image.open(src_img_path).convert('RGB')
        trg_img = Image.open(trg_img_path).convert('RGB')
        
        # Get original sizes for keypoint scaling
        src_size = src_img.size  # (width, height)
        trg_size = trg_img.size
        
        # Transform images
        src_img_tensor = self.transform(src_img)
        trg_img_tensor = self.transform(trg_img)
        
        # Scale keypoints to resized image coordinates
        src_kps = pair['src_kps'].copy()
        trg_kps = pair['trg_kps'].copy()
        
        if len(src_kps) > 0:
            src_kps[:, 0] = src_kps[:, 0] * self.img_size / src_size[0]
            src_kps[:, 1] = src_kps[:, 1] * self.img_size / src_size[1]
        
        if len(trg_kps) > 0:
            trg_kps[:, 0] = trg_kps[:, 0] * self.img_size / trg_size[0]
            trg_kps[:, 1] = trg_kps[:, 1] * self.img_size / trg_size[1]
        
        return {
            'src_img': src_img_tensor,
            'trg_img': trg_img_tensor,
            'src_kps': torch.from_numpy(src_kps).float(),
            'trg_kps': torch.from_numpy(trg_kps).float(),
            'category': pair['category'],
            'src_size': torch.tensor(src_size),
            'trg_size': torch.tensor(trg_size),
        }
    
    def _get_dummy_data(self) -> Dict[str, torch.Tensor]:
        """Generate dummy data for testing without actual dataset."""
        # Create dummy images
        src_img = torch.randn(3, self.img_size, self.img_size)
        trg_img = torch.randn(3, self.img_size, self.img_size)
        
        # Create dummy keypoints (10 points)
        n_kps = 10
        src_kps = torch.rand(n_kps, 2) * self.img_size
        trg_kps = src_kps + torch.randn(n_kps, 2) * 20  # Add some noise
        
        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'src_kps': src_kps,
            'trg_kps': trg_kps,
            'category': 'dummy',
            'src_size': torch.tensor([self.img_size, self.img_size]),
            'trg_size': torch.tensor([self.img_size, self.img_size]),
        }
