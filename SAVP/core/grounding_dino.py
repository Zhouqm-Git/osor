import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


class GroundingDINODetector:
    
    def __init__(self, config_file: str, checkpoint: str, 
                 device: str = "cuda:0", score_threshold: float = 0.3):
        self.score_threshold = score_threshold
        
        mmdet_root = Path(__file__).parent.parent / "mmdetection"
        if mmdet_root.exists() and str(mmdet_root) not in sys.path:
            sys.path.insert(0, str(mmdet_root))
        
        from mmdet.apis import DetInferencer
        
        self.inferencer = DetInferencer(
            model=config_file,
            weights=checkpoint,
            device=device,
            show_progress=False
        )
    
    def detect(self, image: np.ndarray, prompt: str, 
               top_k: Optional[int] = None) -> List[Tuple[int, int, int, int, float]]:
        results = self.inferencer(
            inputs=image,
            texts=prompt,
            tokens_positive=-1,
            pred_score_thr=self.score_threshold,
            return_datasamples=True,
            no_save_vis=True,
            no_save_pred=True,
            show=False
        )
        
        predictions = results['predictions']
        if not predictions:
            return []
        
        pred = predictions[0]
        bboxes = pred.pred_instances.bboxes.cpu().numpy()
        scores = pred.pred_instances.scores.cpu().numpy()
        
        detections = []
        for bbox, score in zip(bboxes, scores):
            x1, y1, x2, y2 = map(int, bbox)
            detections.append((x1, y1, x2, y2, float(score)))
        
        detections.sort(key=lambda x: x[4], reverse=True)
        
        if top_k is not None and len(detections) > top_k:
            detections = detections[:top_k]
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray], prompts: List[str], 
                     top_k: Optional[int] = None) -> List[List[Tuple[int, int, int, int, float]]]:
        results = self.inferencer(
            inputs=images,
            texts=prompts,
            tokens_positive=[-1] * len(images),
            pred_score_thr=self.score_threshold,
            return_datasamples=True,
            no_save_vis=True,
            no_save_pred=True,
            show=False,
            batch_size=len(images)
        )
        
        all_detections = []
        predictions = results['predictions']
        
        for pred in predictions:
            bboxes = pred.pred_instances.bboxes.cpu().numpy()
            scores = pred.pred_instances.scores.cpu().numpy()
            
            detections = []
            for bbox, score in zip(bboxes, scores):
                x1, y1, x2, y2 = map(int, bbox)
                detections.append((x1, y1, x2, y2, float(score)))
            
            detections.sort(key=lambda x: x[4], reverse=True)
            
            if top_k is not None and len(detections) > top_k:
                detections = detections[:top_k]
            
            all_detections.append(detections)
        
        return all_detections

