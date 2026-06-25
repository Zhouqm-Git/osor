from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    kept_bboxes: List[Tuple[int, int, int, int]]


class GeometryValidator:
    
    def __init__(self, iou_threshold: float = 0.3, scale_threshold: float = 2.5):
        self.iou_threshold = iou_threshold
        self.scale_threshold = scale_threshold
    
    def compute_iou(self, bbox1: Tuple[int, int, int, int], 
                    bbox2: Tuple[int, int, int, int]) -> float:
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def compute_area(self, bbox: Tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def validate_diff_with_semantic(
        self, 
        diff_bboxes: List[Tuple[int, int, int, int]], 
        semantic_bboxes: List[Tuple[int, int, int, int, float]]
    ) -> ValidationResult:
        
        if not diff_bboxes:
            return ValidationResult(
                is_valid=False,
                reason="no_diff_regions",
                kept_bboxes=[]
            )
        
        if not semantic_bboxes:
            return ValidationResult(
                is_valid=False,
                reason="no_semantic_targets",
                kept_bboxes=[]
            )
        
        kept_bboxes = []
        
        for diff_bbox in diff_bboxes:
            diff_area = self.compute_area(diff_bbox)
            
            max_iou = 0.0
            max_iou_sem_bbox = None
            
            for sem_bbox_with_score in semantic_bboxes:
                sem_bbox = sem_bbox_with_score[:4]
                iou = self.compute_iou(diff_bbox, sem_bbox)
                if iou > max_iou:
                    max_iou = iou
                    max_iou_sem_bbox = sem_bbox
            
            if max_iou >= self.iou_threshold:
                sem_area = self.compute_area(max_iou_sem_bbox)
                area_ratio = diff_area / max(sem_area, 1)
                
                if area_ratio > self.scale_threshold:
                    return ValidationResult(
                        is_valid=False,
                        reason="scale_overflow_aligned",
                        kept_bboxes=[]
                    )
                
                kept_bboxes.append(diff_bbox)
            
            else:
                if max_iou_sem_bbox is not None:
                    sem_area = self.compute_area(max_iou_sem_bbox)
                    area_ratio = diff_area / max(sem_area, 1)
                    
                    if area_ratio > self.scale_threshold:
                        return ValidationResult(
                            is_valid=False,
                            reason="background_collapse",
                            kept_bboxes=[]
                        )
        
        if not kept_bboxes:
            return ValidationResult(
                is_valid=False,
                reason="no_valid_regions",
                kept_bboxes=[]
            )
        
        return ValidationResult(
            is_valid=True,
            reason="pass",
            kept_bboxes=kept_bboxes
        )

