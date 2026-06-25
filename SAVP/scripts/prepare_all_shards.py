import argparse
import os
import json
import glob
import multiprocessing
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from prepare_dataset import prepare_single_shard


BAD_FILES = {
    "data-00010-of-00037.parquet",
    "data-00024-of-00037.parquet",
    "data-00030-of-00037.parquet",
    "data-00031-of-00037.parquet"
}


def get_shard_stats(filepath):
    try:
        table = pq.read_table(filepath, columns=['category'])
        categories = table['category'].to_pylist()
        
        n_add = categories.count('Add object')
        n_remove = categories.count('Remove object')
        return filepath, n_add, n_remove
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return filepath, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Prepare all NHR-Edit add/remove shards for SAVP.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing NHR-Edit parquet shards.")
    parser.add_argument("--output-root", default=Path("demo"), type=Path, help="Output directory for bg/, shot/, and prompts.json.")
    parser.add_argument("--num-workers", default=32, type=int, help="CPU workers per shard.")
    parser.add_argument("--scan-workers", default=16, type=int, help="Threads used to scan shard categories.")
    parser.add_argument("--num-shards", default=37, type=int, help="Total shard count used in parquet filenames.")
    parser.add_argument("--hf-cache-dir", type=Path, help="Optional Hugging Face datasets cache directory.")
    args = parser.parse_args()

    if args.hf_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = str(args.hf_cache_dir)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    all_files = sorted(glob.glob(str(args.data_dir / "*.parquet")))
    good_files = [f for f in all_files if os.path.basename(f) not in BAD_FILES]
    
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.scan_workers) as executor:
        scan_results = list(tqdm(executor.map(get_shard_stats, good_files), 
                                total=len(good_files), desc="Scanning"))
    
    total_adds_global = sum(r[1] for r in scan_results)
    total_removes_global = sum(r[2] for r in scan_results)
    
    print(f"Total: {total_adds_global} Add + {total_removes_global} Remove = {total_adds_global + total_removes_global}")
    print(f"Add IDs: 0-{total_adds_global-1}, Remove IDs: {total_adds_global}-{total_adds_global+total_removes_global-1}")
    
    current_add_id = 0
    current_remove_id = total_adds_global
    all_prompts = {}
    
    for i, (fpath, n_add, n_remove) in enumerate(scan_results):
        shard_id = int(Path(fpath).stem.split('-')[1])
        
        total_add_processed, total_remove_processed, shard_prompts = prepare_single_shard(
            data_dir=args.data_dir,
            output_root=args.output_root,
            shard_id=shard_id,
            num_workers=args.num_workers,
            start_id_add=current_add_id,
            start_id_remove=current_remove_id,
            num_shards=args.num_shards,
        )
        
        all_prompts.update(shard_prompts)
        current_add_id += total_add_processed
        current_remove_id += total_remove_processed
    
    prompt_file = args.output_root / "prompts.json"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prompt_file, 'w', encoding='utf-8') as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted: {len(all_prompts)} samples saved to {prompt_file}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
