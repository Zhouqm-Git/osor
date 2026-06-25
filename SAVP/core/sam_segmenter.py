import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


class SAM2Segmenter:
    
    def __init__(self, model_cfg: str, checkpoint: str, device: str = "cuda:0"):
        sam2_root = Path(__file__).parent.parent / "sam2"
        if sam2_root.exists() and str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        self.device = device
        model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(model)
    
    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)
    
    def predict_with_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        if not boxes:
            return []
        
        boxes_np = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2 in boxes])
        
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False
        )
        
        mask_list = []
        for i in range(len(boxes)):
            if masks.ndim == 3:
                mask_list.append(masks[i].astype(np.uint8))
            else:
                mask_list.append(masks.astype(np.uint8))
        
        return mask_list
    
    def predict_batch(self, images: List[np.ndarray], 
                      boxes_list: List[List[Tuple[int, int, int, int]]]) -> List[List[np.ndarray]]:
        all_masks = []
        for image, boxes in zip(images, boxes_list):
            self.set_image(image)
            masks = self.predict_with_boxes(boxes)
            all_masks.append(masks)
        return all_masks

