from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Any

import numpy as np
import cv2
from skimage import morphology
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt, binary_fill_holes


@dataclass
class QualityMetrics:
    is_hq: bool
    reason: str
    main_count: Optional[int] = None
    noise_ratio: Optional[float] = None

class FeatureEngine:
    """Handles low-level image difference calculations"""
    
    @staticmethod
    def _compute_features(Iin: np.ndarray, Igt: np.ndarray) -> Dict[str, np.ndarray]:
        # Pre-process: Gamma correction / Normalization
        rgb_in = (Iin.astype(np.float32) / 255.0) ** 2.2
        rgb_gt = (Igt.astype(np.float32) / 255.0) ** 2.2
        
        # Convert to LAB
        lab_in = cv2.cvtColor(np.clip(rgb_in * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
        lab_gt = cv2.cvtColor(np.clip(rgb_gt * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2Lab).astype(np.float32)
        
        # Extract channels
        L_in, L_gt = lab_in[..., 0] * (100/255), lab_gt[..., 0] * (100/255)
        ab_in, ab_gt = lab_in[..., 1:] - 128.0, lab_gt[..., 1:] - 128.0
        
        # 1. Log-Luminance Difference
        eps = 1e-3
        dL = np.abs(np.log(L_gt + eps) - np.log(L_in + eps))
        
        # 2. Chromaticity Difference (Euclidean distance in ab plane)
        dab = np.linalg.norm(ab_gt - ab_in, axis=-1)
        
        # 3. Texture Difference (Sobel Magnitude)
        gray_in = cv2.cvtColor(Iin, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gray_gt = cv2.cvtColor(Igt, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        def get_sobel(g):
            sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            return np.hypot(sx, sy)
            
        dtex = np.abs(get_sobel(gray_gt) - get_sobel(gray_in))
        
        # Normalization helper
        def normalize(x):
            if np.any(x > 0):
                p99 = np.percentile(x[x > 0], 99)
                return np.clip(x / (p99 + eps), 0, 1)
            return np.zeros_like(x)
            
        return {
            'dL': normalize(dL), 
            'dab': normalize(dab), 
            'dtex': normalize(dtex)
        }

    @staticmethod
    def generate_heatmap(Iin: np.ndarray, Igt: np.ndarray, w_l: float = 0.6, 
                        w_ab: float = 0.3, w_tex: float = 0.1) -> np.ndarray:
        feats = FeatureEngine._compute_features(Iin, Igt)
        heatmap = (w_l * feats['dL']) + (w_ab * feats['dab']) + (w_tex * feats['dtex'])
        return np.clip(heatmap, 0, 1)


class MorphologyEngine:
    
    def __init__(self, min_area: int = 2000, max_hole_area: int = 500):
        self.min_area = min_area
        self.max_hole_area = max_hole_area

    def clean_raw_mask(self, mask: np.ndarray) -> np.ndarray:
        # 1. Morphological Open/Close
        selem = morphology.disk(3)
        mask = morphology.closing(morphology.opening(mask > 0, selem), selem)
        
        # 2. Remove Small Components
        labeled = label(mask)
        out = np.zeros_like(mask, dtype=np.uint8)
        for prop in regionprops(labeled):
            if prop.area >= self.min_area:
                out[labeled == prop.label] = 1
        
        # 3. Fill Internal Holes
        filled = binary_fill_holes(out)
        holes = filled.astype(np.uint8) - out
        for prop in regionprops(label(holes)):
            if prop.area <= self.max_hole_area:
                out[label(holes) == prop.label] = 1
                
        return out.astype(np.uint8)

    def expand_mask(self, mask: np.ndarray, ratio: float) -> np.ndarray:
        """Expands mask by a ratio using distance transform"""
        if ratio <= 1.0 or np.count_nonzero(mask) == 0:
            return mask
            
        current_area = np.count_nonzero(mask)
        target_area = int(current_area * ratio)
        pixels_to_add = target_area - current_area
        
        if pixels_to_add <= 0: return mask
        
        dist_map = distance_transform_edt(mask == 0)
        bg_dists = dist_map[mask == 0]
        
        if bg_dists.size == 0: return mask
        
        # Determine threshold to add exactly k pixels
        if pixels_to_add >= bg_dists.size:
            return np.ones_like(mask)
            
        k = pixels_to_add
        # Optimization: np.partition is O(n) vs sort O(n log n)
        threshold = np.partition(bg_dists, k - 1)[k - 1]
        
        expanded = (mask > 0) | (dist_map <= threshold)
        return expanded.astype(np.uint8)

    def compute_soft_edge(self, mask: np.ndarray, feather_px: int = 8) -> np.ndarray:
        if not mask.any(): 
            return mask.astype(np.float32)
        dist = distance_transform_edt(~mask.astype(bool))
        alpha = np.where(mask, 1.0, np.clip(1.0 - dist / max(1, feather_px), 0, 1))
        return alpha.astype(np.float32)


class QualityAuditor:
    
    def __init__(self, noise_threshold: float = 0.3, max_main_regions: int = 5,
                 area_threshold_ratio: float = 0.3):
        self.noise_threshold = noise_threshold
        self.max_main_regions = max_main_regions
        self.area_threshold_ratio = area_threshold_ratio

    def audit(self, mask: np.ndarray) -> QualityMetrics:
        labeled = label(mask > 0)
        props = regionprops(labeled)
        
        if not props:
            return QualityMetrics(is_hq=False, reason="empty_mask")
        
        props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
        max_area = props_sorted[0].area
        
        split_thresh = max_area * self.area_threshold_ratio
        main_components = [p for p in props_sorted if p.area >= split_thresh]
        noise_components = [p for p in props_sorted if p.area < split_thresh]
        
        if len(main_components) > self.max_main_regions:
            return QualityMetrics(
                is_hq=False, 
                reason="too_many_objects",
                main_count=len(main_components)
            )
            
        total_area = mask.sum()
        noise_area = sum(p.area for p in noise_components)
        noise_ratio = noise_area / max(total_area, 1)
        
        if noise_ratio > self.noise_threshold:
            return QualityMetrics(
                is_hq=False,
                reason="high_noise",
                noise_ratio=float(f"{noise_ratio:.3f}")
            )
            
        return QualityMetrics(
            is_hq=True,
            reason="pass",
            main_count=len(main_components)
        )


class DifferenceEngine:
    
    def __init__(self, threshold: float = 0.07, min_area: int = 2000,
                 max_hole_area: int = 500, w_l: float = 0.6, 
                 w_ab: float = 0.3, w_tex: float = 0.1,
                 noise_threshold: float = 0.3, max_main_regions: int = 5,
                 area_threshold_ratio: float = 0.3):
        self.threshold = threshold
        self.w_l = w_l
        self.w_ab = w_ab
        self.w_tex = w_tex
        self.morph_engine = MorphologyEngine(min_area, max_hole_area)
        self.auditor = QualityAuditor(noise_threshold, max_main_regions, area_threshold_ratio)
    
    def compute_diff_mask(self, img_bg: np.ndarray, 
                         img_shot: np.ndarray) -> Tuple[np.ndarray, np.ndarray, QualityMetrics]:
        heatmap = FeatureEngine.generate_heatmap(img_bg, img_shot, 
                                                 self.w_l, self.w_ab, self.w_tex)
        raw_mask = (heatmap > self.threshold).astype(np.uint8)
        clean_mask = self.morph_engine.clean_raw_mask(raw_mask)
        metrics = self.auditor.audit(clean_mask)
        return clean_mask, heatmap, metrics
    
    def get_main_diff_regions(self, mask: np.ndarray, 
                              area_threshold_ratio: float = 0.3) -> List[Tuple]:
        labeled = label(mask > 0)
        props = regionprops(labeled)
        
        if not props:
            return []
        
        props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
        max_area = props_sorted[0].area
        split_thresh = max_area * area_threshold_ratio
        
        main_regions = []
        for prop in props_sorted:
            if prop.area >= split_thresh:
                bbox = prop.bbox
                main_regions.append((bbox[1], bbox[0], bbox[3], bbox[2], prop.area))
        
        return main_regions