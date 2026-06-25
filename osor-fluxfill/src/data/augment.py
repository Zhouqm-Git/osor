import cv2
import numpy as np
import random
from PIL import Image

class MaskAugmentor:
    def __init__(self, 
                 shrink_ratio_range=(0.5, 0.7), 
                 dilate_range=(1, 3), 
                 shift_limit=50,
                 expand_kernels=[11, 17, 21],
                 hole_kernels=[21, 31, 41]):
        self.shrink_ratio_range = shrink_ratio_range
        self.dilate_range = dilate_range
        self.shift_limit = shift_limit
        self.expand_kernels = expand_kernels
        self.hole_kernels = hole_kernels

    def _to_numpy(self, img):
        arr = np.array(img)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return (arr > 128).astype(np.uint8) * 255

    def shrink_by_ratio(self, mask_np):
        current_area = np.count_nonzero(mask_np)
        if current_area == 0: return mask_np
        
        ratio = random.uniform(*self.shrink_ratio_range)
        target_area = int(current_area * ratio)
        pixels_to_remove = current_area - target_area
        
        if pixels_to_remove <= 0: return mask_np

        dist_map = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5)
        fg_dists = dist_map[mask_np > 0]
        
        k = pixels_to_remove
        threshold = np.partition(fg_dists, k - 1)[k - 1]
        
        new_mask = np.zeros_like(mask_np)
        new_mask[dist_map > threshold] = 255
        return new_mask

    def convex_expand(self, mask_np):
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return mask_np
        
        hull_mask = np.zeros_like(mask_np)
        hulls = [cv2.convexHull(c) for c in contours]
        cv2.drawContours(hull_mask, hulls, -1, 255, -1)
        
        k = random.choice(self.expand_kernels)
        kernel = np.ones((k, k), np.uint8)
        iter_n = random.randint(*self.dilate_range)
        return cv2.dilate(hull_mask, kernel, iterations=iter_n)

    def random_holes(self, mask_np):
        h, w = mask_np.shape
        seeds = (np.random.random((h, w)) < 0.0005).astype(np.uint8) * 255
        
        k = random.choice(self.hole_kernels)
        kernel = np.ones((k, k), np.uint8)
        holes = cv2.dilate(seeds, kernel, iterations=1)
        
        aug_mask = cv2.bitwise_and(mask_np, cv2.bitwise_not(holes))
        
        if np.sum(aug_mask) < np.sum(mask_np) * 0.4:
            return mask_np
        return aug_mask

    def random_shift(self, mask_np):
        dx = random.randint(-self.shift_limit, self.shift_limit)
        dy = random.randint(-self.shift_limit, self.shift_limit)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(mask_np, M, (mask_np.shape[1], mask_np.shape[0]), borderValue=0)

    def __call__(self, mask_img, mode='shrink', sam_mask=None):
        if mode == 'sam' and sam_mask is not None:
            return sam_mask

        mask_np = self._to_numpy(mask_img)
        
        if mode == 'shrink':
            res = self.shrink_by_ratio(mask_np)
        elif mode == 'expand':
            res = self.convex_expand(mask_np)
        elif mode == 'holes':
            res = self.random_holes(mask_np)
        elif mode == 'shift':
            res = self.random_shift(mask_np)
        else:
            res = mask_np
            
        return Image.fromarray(res).convert("L")