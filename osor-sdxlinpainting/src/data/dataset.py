import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class ObjectRemovalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        shot_list: str,
        mask_list: str,
        bg_list: str,
        augment_mask: bool = False,
        mask_sam_list: str = None,
        resolution: int = 512,
    ):
        self.data_root = Path(data_root)
        self.resolution = resolution
        self.augment_mask = augment_mask

        self.shot_paths = self._load_file_list(shot_list)
        self.mask_paths = self._load_file_list(mask_list)
        self.bg_paths = self._load_file_list(bg_list)

        self.mask_sam_paths = None
        if mask_sam_list:
            self.mask_sam_paths = self._load_file_list(mask_sam_list)
            if len(self.mask_sam_paths) != len(self.shot_paths):
                raise ValueError("SAM mask list length mismatch")

        if not (len(self.shot_paths) == len(self.mask_paths) == len(self.bg_paths)):
            raise ValueError("Lists length mismatch")
            
        self.resolutions = []
        self._cache_resolutions()

    def _load_file_list(self, file_list_path: str) -> List[str]:
        paths = []
        with open(file_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if not os.path.isabs(line):
                    line = str(self.data_root / line)
                paths.append(line)
        return paths

    def _cache_resolutions(self):
        print("Caching resolutions...")
        for path in self.shot_paths:
            try:
                with Image.open(path) as img:
                    self.resolutions.append(img.size)
            except:
                self.resolutions.append((512, 512))
        print("Done.")

    def get_resolution(self, idx):
        return self.resolutions[idx]

    def _calculate_target_size(self, width: int, height: int) -> Tuple[int, int]:
        target = self.resolution
        min_dim = min(width, height)
        if min_dim < target:
            scale = target / min_dim
            w, h = int(width * scale), int(height * scale)
        else:
            if width < height:
                w, h = target, int(height * target / width)
            else:
                h, w = target, int(width * target / height)
        return (w + 15) // 16 * 16, (h + 15) // 16 * 16

    def _load_image(self, path: str, size: Tuple[int, int]) -> np.ndarray:
        img = Image.open(path).convert('RGB')
        img = img.resize(size, Image.LANCZOS)
        return np.array(img)

    def _load_mask_pil(self, path: str, size: Tuple[int, int]) -> Image.Image:
        img = Image.open(path).convert('L')
        img = img.resize(size, Image.NEAREST)
        return img

    def __len__(self):
        return len(self.shot_paths)

    def __getitem__(self, idx):
        # 1. Resolution & Size
        try:
            orig_w, orig_h = self.resolutions[idx]
        except:
            with Image.open(self.shot_paths[idx]) as img:
                orig_w, orig_h = img.size
        target_size = self._calculate_target_size(orig_w, orig_h)
        
        # 2. Load Images
        shot = self._load_image(self.shot_paths[idx], target_size)
        bg = self._load_image(self.bg_paths[idx], target_size)
        
        # 3. Mask Handling
        mask_pil = self._load_mask_pil(self.mask_paths[idx], target_size)
        mask_np = np.array(mask_pil)
        
        data = {
            'shot': torch.from_numpy(shot).permute(2, 0, 1).float() / 255.0,
            'mask': torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0,
            'bg': torch.from_numpy(bg).permute(2, 0, 1).float() / 255.0,
            'shot_path': self.shot_paths[idx],
            'mask_path': self.mask_paths[idx],
            'bg_path': self.bg_paths[idx]
        }

        # Inject PIL object ONLY if augmentation is enabled
        if self.augment_mask:
            data['_raw_mask_pil'] = mask_pil
            if self.mask_sam_paths:
                sam_mask_pil = self._load_mask_pil(self.mask_sam_paths[idx], target_size)
                data['_sam_mask_pil'] = sam_mask_pil

        return data