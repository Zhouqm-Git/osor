import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import PipelineConfig
from core.pipeline import EditMaskPipeline


def load_dataset(bg_dir: Path, shot_dir: Path, 
                 prompt_file: Path = None) -> List[Tuple[Path, Path, str]]:
    valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    bg_files = sorted([f for f in bg_dir.iterdir() if f.suffix in valid_exts])
    shot_files = sorted([f for f in shot_dir.iterdir() if f.suffix in valid_exts])
    
    bg_map = {f.stem: f for f in bg_files}
    shot_map = {f.stem: f for f in shot_files}
    
    common_stems = set(bg_map.keys()) & set(shot_map.keys())
    
    prompts = {}
    if prompt_file and prompt_file.exists():
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
    
    dataset = []
    for stem in sorted(common_stems):
        prompt = prompts.get(stem, "edited region")
        dataset.append((bg_map[stem], shot_map[stem], prompt))
    
    return dataset


def process_worker(pipeline: EditMaskPipeline, dataset: List[Tuple], 
                   output_dir: Path, gpu_id: int, worker_index: int, 
                   start_idx: int, end_idx: int, log_interval: int = 100):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"worker_{worker_index}_gpu_{gpu_id}_log.txt"
    stats = {"success": 0, "failed": 0, "reasons": {}}
    
    with open(log_file, 'w') as log:
        log.write(f"Worker {worker_index} on GPU {gpu_id}: Processing indices [{start_idx}, {end_idx})\n")
        log.write(f"Total samples: {end_idx - start_idx}\n\n")
        
        for idx in tqdm(range(start_idx, end_idx), desc=f"Worker {worker_index} (GPU {gpu_id})"):
            bg_path, shot_path, prompt = dataset[idx]
            stem = bg_path.stem
            
            try:
                result = pipeline.process_from_paths(
                    str(bg_path), str(shot_path), prompt
                )
                
                if result.is_valid:
                    mask_path = output_dir / "masks" / f"mask_{stem}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), result.final_mask * 255)
                    stats["success"] += 1
                    
                    if (idx - start_idx + 1) % log_interval == 0:
                        log.write(f"[{idx - start_idx + 1}/{end_idx - start_idx}] ✓ {stem}\n")
                        log.flush()
                else:
                    stats["failed"] += 1
                    reason = result.reason
                    stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                    log.write(f"✗ {stem}: {reason}\n")
                    log.flush()
                    
            except Exception as e:
                stats["failed"] += 1
                stats["reasons"]["error"] = stats["reasons"].get("error", 0) + 1
                log.write(f"✗ {stem}: Exception - {str(e)}\n")
                log.flush()
        
        log.write(f"\n{'='*60}\n")
        log.write(f"Summary for Worker {worker_index} (GPU {gpu_id}):\n")
        log.write(f"  Success: {stats['success']}\n")
        log.write(f"  Failed: {stats['failed']}\n")
        log.write(f"  Failure breakdown:\n")
        for reason, count in stats["reasons"].items():
            log.write(f"    - {reason}: {count}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="GPU Worker for Edit Mask Pipeline")
    parser.add_argument('--bg-dir', required=True, type=Path, help='Background images directory')
    parser.add_argument('--shot-dir', required=True, type=Path, help='Shot images directory')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prompt-file', required=True, type=Path, help='JSON file: {stem: prompt}')
    
    parser.add_argument('--gpu-id', type=int, required=True, help='Physical GPU ID to use')
    parser.add_argument('--total-workers', type=int, required=True, help='Total number of worker processes')
    parser.add_argument('--worker-index', type=int, required=True, help='Current worker index (0-based)')
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N samples')
    
    parser.add_argument('--dino-config', type=str, help='Override DINO config path')
    parser.add_argument('--dino-checkpoint', type=str, help='Override DINO checkpoint path')
    parser.add_argument('--sam-config', type=str, help='Override SAM config')
    parser.add_argument('--sam-checkpoint', type=str, help='Override SAM checkpoint path')
    
    args = parser.parse_args()
    
    dataset = load_dataset(args.bg_dir, args.shot_dir, args.prompt_file)
    total_samples = len(dataset)
    print(f"Loaded {total_samples} image pairs")
    
    samples_per_worker = total_samples // args.total_workers
    remainder = total_samples % args.total_workers
    
    start_idx = args.worker_index * samples_per_worker + min(args.worker_index, remainder)
    if args.worker_index < remainder:
        end_idx = start_idx + samples_per_worker + 1
    else:
        end_idx = start_idx + samples_per_worker
    
    print(f"Worker {args.worker_index} on GPU {args.gpu_id} processing indices [{start_idx}, {end_idx})")
    
    config = PipelineConfig()
    config.grounding_dino.device = f"cuda:{args.gpu_id}"
    config.sam2.device = f"cuda:{args.gpu_id}"
    
    if args.dino_config:
        config.grounding_dino.config_file = args.dino_config
    if args.dino_checkpoint:
        config.grounding_dino.checkpoint = args.dino_checkpoint
    if args.sam_config:
        config.sam2.model_cfg = args.sam_config
    if args.sam_checkpoint:
        config.sam2.checkpoint = args.sam_checkpoint
    
    pipeline = EditMaskPipeline(config)
    
    stats = process_worker(pipeline, dataset, args.output_dir, 
                          args.gpu_id, args.worker_index, start_idx, end_idx, args.log_interval)
    
    print(f"\nWorker {args.worker_index} (GPU {args.gpu_id}) finished:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")


if __name__ == "__main__":
    main()

