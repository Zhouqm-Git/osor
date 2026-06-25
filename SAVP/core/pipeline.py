from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import cv2

from .config import PipelineConfig
from .difference import DifferenceEngine, QualityMetrics
from .grounding_dino import GroundingDINODetector
from .sam_segmenter import SAM2Segmenter
from .geometry_validator import GeometryValidator, ValidationResult


@dataclass
class PipelineResult:
    is_valid: bool
    reason: str
    final_mask: Optional[np.ndarray] = None
    quality_metrics: Optional[QualityMetrics] = None
    validation_result: Optional[ValidationResult] = None
    semantic_bboxes: Optional[list] = None
    diff_bboxes: Optional[list] = None
    diff_mask_pre_fusion: Optional[np.ndarray] = None


class EditMaskPipeline:
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        if config is None:
            config = PipelineConfig()
        
        self.config = config
        
        self.diff_engine = DifferenceEngine(
            threshold=config.difference.threshold,
            min_area=config.difference.min_area,
            max_hole_area=config.difference.max_hole_area,
            w_l=config.difference.w_l,
            w_ab=config.difference.w_ab,
            w_tex=config.difference.w_tex,
            noise_threshold=config.difference.noise_threshold,
            max_main_regions=config.difference.max_main_regions,
            area_threshold_ratio=config.difference.area_threshold_ratio
        )
        
        self.dino = GroundingDINODetector(
            config_file=config.grounding_dino.config_file,
            checkpoint=config.grounding_dino.checkpoint,
            device=config.grounding_dino.device,
            score_threshold=config.grounding_dino.score_threshold
        )
        
        self.sam = SAM2Segmenter(
            model_cfg=config.sam2.model_cfg,
            checkpoint=config.sam2.checkpoint,
            device=config.sam2.device
        )
        
        self.validator = GeometryValidator(
            iou_threshold=config.geometry.iou_threshold,
            scale_threshold=config.geometry.scale_threshold
        )
    
    def process(self, img_bg: np.ndarray, img_shot: np.ndarray, 
                prompt: str) -> PipelineResult:
        
        diff_mask, heatmap, quality = self.diff_engine.compute_diff_mask(
            img_bg, img_shot
        )
        
        if not quality.is_hq:
            return PipelineResult(
                is_valid=False,
                reason=f"diff_quality_{quality.reason}",
                quality_metrics=quality,
                diff_mask_pre_fusion=diff_mask
            )
        
        diff_regions = self.diff_engine.get_main_diff_regions(
            diff_mask, 
            self.config.difference.area_threshold_ratio
        )
        
        if not diff_regions:
            return PipelineResult(
                is_valid=False,
                reason="no_diff_regions",
                quality_metrics=quality,
                diff_mask_pre_fusion=diff_mask
            )
        
        diff_bboxes = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in diff_regions]
        diff_regions_map = {(x1, y1, x2, y2): region for region in diff_regions 
                           for x1, y1, x2, y2, _ in [region]}
        
        semantic_bboxes = self.dino.detect(
            img_shot, 
            prompt, 
            top_k=self.config.grounding_dino.top_k
        )
        
        validation = self.validator.validate_diff_with_semantic(
            diff_bboxes, 
            semantic_bboxes
        )
        
        if not validation.is_valid:
            return PipelineResult(
                is_valid=False,
                reason=validation.reason,
                quality_metrics=quality,
                validation_result=validation,
                semantic_bboxes=semantic_bboxes,
                diff_bboxes=diff_bboxes,
                diff_mask_pre_fusion=diff_mask
            )
        
        self.sam.set_image(img_shot)
        sam_masks = self.sam.predict_with_boxes(validation.kept_bboxes)
        
        filtered_diff_mask = self._create_filtered_diff_mask(
            diff_mask, validation.kept_bboxes, img_shot.shape[:2]
        )
        
        final_mask = self._fuse_masks(filtered_diff_mask, sam_masks, img_shot.shape[:2])
        
        if self.config.mask.expand_ratio > 1.0:
            final_mask = self.diff_engine.morph_engine.expand_mask(
                final_mask, 
                self.config.mask.expand_ratio
            )
        
        return PipelineResult(
            is_valid=True,
            reason="success",
            final_mask=final_mask,
            quality_metrics=quality,
            validation_result=validation,
            semantic_bboxes=semantic_bboxes,
            diff_bboxes=diff_bboxes,
            diff_mask_pre_fusion=filtered_diff_mask
        )
    
    def _create_filtered_diff_mask(self, diff_mask: np.ndarray, 
                                   kept_bboxes: list, 
                                   shape: Tuple[int, int]) -> np.ndarray:
        from skimage.measure import label, regionprops
        
        h, w = shape
        filtered = np.zeros((h, w), dtype=np.uint8)
        
        labeled = label(diff_mask > 0)
        props = regionprops(labeled)
        
        for prop in props:
            bbox_prop = prop.bbox
            x1, y1, x2, y2 = bbox_prop[1], bbox_prop[0], bbox_prop[3], bbox_prop[2]
            
            for kept_bbox in kept_bboxes:
                kx1, ky1, kx2, ky2 = kept_bbox
                
                xi1 = max(x1, kx1)
                yi1 = max(y1, ky1)
                xi2 = min(x2, kx2)
                yi2 = min(y2, ky2)
                
                if xi2 > xi1 and yi2 > yi1:
                    inter_area = (xi2 - xi1) * (yi2 - yi1)
                    prop_area = prop.area
                    
                    if inter_area / prop_area > 0.5:
                        filtered[labeled == prop.label] = 1
                        break
        
        return filtered
    
    def _fuse_masks(self, diff_mask: np.ndarray, sam_masks: list, 
                    shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        final = np.zeros((h, w), dtype=np.uint8)
        
        for mask in sam_masks:
            final = np.logical_or(final, mask > 0)
        
        final = np.logical_or(final, diff_mask > 0)
        
        return final.astype(np.uint8)
    
    def process_from_paths(self, img_bg_path: str, img_shot_path: str, 
                          prompt: str) -> PipelineResult:
        img_bg = cv2.cvtColor(cv2.imread(img_bg_path), cv2.COLOR_BGR2RGB)
        img_shot = cv2.cvtColor(cv2.imread(img_shot_path), cv2.COLOR_BGR2RGB)
        
        if img_bg is None or img_shot is None:
            return PipelineResult(
                is_valid=False,
                reason="image_load_error"
            )
        
        if img_bg.shape != img_shot.shape:
            return PipelineResult(
                is_valid=False,
                reason="shape_mismatch"
            )
        
        return self.process(img_bg, img_shot, prompt)


SAVPPipeline = EditMaskPipeline
