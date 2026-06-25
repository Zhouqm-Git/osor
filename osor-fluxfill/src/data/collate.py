import torch
import numpy as np
import random
from torch.utils.data import default_collate
from .augment import MaskAugmentor

class ConsistentMaskCollate:
    def __init__(self):
        self.augmentor = MaskAugmentor()

    def __call__(self, batch):
        # Compatibility check: if no special key, use default behavior
        if not batch or '_raw_mask_pil' not in batch[0]:
            return default_collate(batch)

        modes = ['shrink', 'expand', 'holes', 'shift', 'sam']
        weights = [0.35, 0.2, 0.1, 0.05, 0.3]
        
        current_batch_mode = random.choices(modes, weights=weights, k=1)[0]

        for item in batch:
            # Retrieve PIL object
            raw_pil = item.pop('_raw_mask_pil')
            sam_pil = item.pop('_sam_mask_pil', None)
            
            # Fallback if 'sam' chosen but not available
            mode_to_use = current_batch_mode
            if mode_to_use == 'sam' and sam_pil is None:
                mode_to_use = 'shrink'
            
            # Generate degraded input mask
            aug_pil = self.augmentor(raw_pil, mode=mode_to_use, sam_mask=sam_pil)
            aug_np = np.array(aug_pil)
            
            # Preserve GT and update input mask
            # Ensure 'gt_mask' exists for loss calculation
            if 'gt_mask' not in item:
                item['gt_mask'] = item['mask'].clone()
            
            # Overwrite 'mask' with degraded version for model input
            item['mask'] = torch.from_numpy(aug_np).unsqueeze(0).float() / 255.0

        return default_collate(batch)