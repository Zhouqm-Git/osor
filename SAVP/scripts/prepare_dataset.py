import argparse
import json
import multiprocessing
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import cv2

def save_image_fast(pil_image, path):
    img_array = np.array(pil_image)
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_array)


def process_batch(args):
    shard_file, indices, start_id, category, bg_dir, shot_dir = args
    
    dataset = load_dataset("parquet", data_files=shard_file, split="train")
    
    all_cats = dataset['category']
    if category == "Add":
        filtered_indices = [i for i in indices if all_cats[i] == 'Add object']
        ds_subset = dataset.select(filtered_indices)
    else:
        filtered_indices = [i for i in indices if all_cats[i] == 'Remove object']
        ds_subset = dataset.select(filtered_indices)
    
    prompts = {}
    
    for i, item in enumerate(ds_subset):
        curr_id = start_id + i
        stem = f"{curr_id:06d}"
        
        bg_path = Path(bg_dir) / f"{stem}.png"
        shot_path = Path(shot_dir) / f"{stem}.png"
        
        if category == "Add":
            save_image_fast(item['source_image'], bg_path)
            save_image_fast(item['edited_image'], shot_path)
        else:
            save_image_fast(item['edited_image'], bg_path)
            save_image_fast(item['source_image'], shot_path)
        
        prompts[stem] = item['edit_instruction']
    
    return prompts


def prepare_single_shard(
    data_dir,
    output_root,
    shard_id=0,
    num_workers=32,
    start_id_add=0,
    start_id_remove=0,
    num_shards=37,
):
    data_dir = Path(data_dir)
    output_root = Path(output_root)
    shard_file = data_dir / f"data-{shard_id:05d}-of-{num_shards:05d}.parquet"
    if not shard_file.exists():
        raise FileNotFoundError(f"Shard not found: {shard_file}")

    dataset = load_dataset("parquet", data_files=shard_file, split="train")
    
    bg_dir = output_root / "bg"
    shot_dir = output_root / "shot"
    bg_dir.mkdir(parents=True, exist_ok=True)
    shot_dir.mkdir(parents=True, exist_ok=True)
    
    all_cats = dataset['category']
    add_indices = [i for i, c in enumerate(all_cats) if c == 'Add object']
    remove_indices = [i for i, c in enumerate(all_cats) if c == 'Remove object']
    
    total_add = len(add_indices)
    total_remove = len(remove_indices)
    
    batch_size_add = max(1, total_add // num_workers)
    batch_size_remove = max(1, total_remove // num_workers)
    
    tasks = []
    
    for i in range(0, total_add, batch_size_add):
        end = min(i + batch_size_add, total_add)
        batch_indices = add_indices[i:end]
        tasks.append((shard_file, batch_indices, start_id_add + i, "Add", str(bg_dir), str(shot_dir)))
    
    for i in range(0, total_remove, batch_size_remove):
        end = min(i + batch_size_remove, total_remove)
        batch_indices = remove_indices[i:end]
        tasks.append((shard_file, batch_indices, start_id_remove + i, "Remove", str(bg_dir), str(shot_dir)))
    
    all_prompts = {}
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap(process_batch, tasks), total=len(tasks), 
                           desc=f"Shard {shard_id}"))
    
    for prompt_dict in results:
        all_prompts.update(prompt_dict)
    
    return total_add, total_remove, all_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NHR-Edit add/remove pairs for SAVP.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing NHR-Edit parquet shards.")
    parser.add_argument("--output-root", default=Path("demo"), type=Path, help="Output directory for bg/, shot/, and prompts.json.")
    parser.add_argument("--shard-id", default=0, type=int, help="Shard index to process.")
    parser.add_argument("--num-shards", default=37, type=int, help="Total shard count used in the parquet filename.")
    parser.add_argument("--num-workers", default=32, type=int, help="Number of CPU workers.")
    parser.add_argument("--hf-cache-dir", type=Path, help="Optional Hugging Face datasets cache directory.")
    args = parser.parse_args()

    if args.hf_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = str(args.hf_cache_dir)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    multiprocessing.set_start_method('spawn', force=True)
    total_add, total_remove, prompts = prepare_single_shard(
        data_dir=args.data_dir,
        output_root=args.output_root,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        num_shards=args.num_shards,
    )
    
    prompt_file = args.output_root / "prompts.json"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    
    print(f"\nShard {args.shard_id}: {total_add} Add + {total_remove} Remove = {len(prompts)} total")
