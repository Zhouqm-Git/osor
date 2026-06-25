import sys
import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from glob import glob

sys.path.append(str(Path(__file__).parent.parent))
from core.sam_segmenter import SAM2Segmenter
from core.config import PipelineConfig
from core.difference import MorphologyEngine

def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols): return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [x_min, y_min, x_max, y_max]

def separate_shadows(data_root, img_dir, mask_dir, out_obj_dir, shadow_thresh, gpu_id, total_workers, worker_index, start_index):
    img_root = os.path.join(data_root, img_dir)
    mask_root = os.path.join(data_root, mask_dir)
    out_path = os.path.join(data_root, out_obj_dir)
    os.makedirs(out_path, exist_ok=True)
    
    config = PipelineConfig()
    sam = SAM2Segmenter(config.sam2.model_cfg, config.sam2.checkpoint, f"cuda:{gpu_id}")
    morph = MorphologyEngine(config.difference.min_area, config.difference.max_hole_area)
    
    mask_files = sorted(glob(os.path.join(mask_root, "*.png")))
    if start_index > 0:
        mask_files = mask_files[start_index:]
    
    total = len(mask_files)
    per_worker = total // total_workers
    rem = total % total_workers
    start = worker_index * per_worker + min(worker_index, rem)
    end = start + per_worker + (1 if worker_index < rem else 0)
    my_files = mask_files[start:end]

    print(f"Worker {worker_index}: Processing {len(my_files)} samples ({start}-{end})")

    for path in tqdm(my_files, desc=f"W{worker_index}"):
        name = os.path.splitext(os.path.basename(path))[0]
        image_stem = name[5:] if name.startswith("mask_") else name
        img_p = next((os.path.join(img_root, image_stem + e) for e in [".png", ".jpg", ".jpeg", ".bmp"] if os.path.exists(os.path.join(img_root, image_stem + e))), None)
        if not img_p: continue

        mask = cv2.imread(path, 0)
        if mask is None: continue
        mask = (mask > 127).astype(np.uint8)
        bbox = get_bbox(mask)
        if not bbox: continue

        img = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        sam.set_image(img)
        preds = sam.predict_with_boxes([bbox])
        if not preds: continue
        
        sam_mask = preds[0]
        expanded = morph.expand_mask(sam_mask, config.mask.expand_ratio)
        shadow_mask = morph.clean_raw_mask(cv2.subtract(mask, expanded))
        
        obj_px = np.count_nonzero(expanded)
        ratio = (np.count_nonzero(shadow_mask) / obj_px) if obj_px > 0 else 0
        
        if ratio > shadow_thresh:
            cv2.imwrite(os.path.join(out_path, image_stem + ".png"), sam_mask * 255)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--img_dir", default="shot")
    p.add_argument("--mask_dir", default="output")
    p.add_argument("--out_obj_dir", default="masks_object")
    p.add_argument("--shadow_thresh", type=float, default=0.15)
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--total_workers", type=int, default=1)
    p.add_argument("--worker_index", type=int, default=0)
    p.add_argument("--start_index", type=int, default=0)
    a = p.parse_args()
    
    separate_shadows(a.data_root, a.img_dir, a.mask_dir, a.out_obj_dir, a.shadow_thresh, a.gpu_id, a.total_workers, a.worker_index, a.start_index)
