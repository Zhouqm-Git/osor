import argparse
import subprocess
import time
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--img-dir', type=str, default='shot')
    parser.add_argument('--mask-dir', type=str, default='output')
    parser.add_argument('--out-obj-dir', type=str, default='masks_object')
    parser.add_argument('--shadow-thresh', type=float, default=0.08)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--gpus', required=True, type=str)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    args = parser.parse_args()
    
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    total_workers = len(gpu_ids) * args.workers_per_gpu
    log_dir = args.data_root / "logs_separation"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Launching {total_workers} workers on GPUs {gpu_ids}. Start Index: {args.start_index}")
    processes = []

    for i in range(total_workers):
        phys_gpu = gpu_ids[i // args.workers_per_gpu]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(phys_gpu)
        
        cmd = [
            'python', 'scripts/separate_shadows.py',
            '--data_root', str(args.data_root),
            '--img_dir', args.img_dir,
            '--mask_dir', args.mask_dir,
            '--out_obj_dir', args.out_obj_dir,
            '--shadow_thresh', str(args.shadow_thresh),
            '--gpu_id', '0',
            '--total_workers', str(total_workers),
            '--worker_index', str(i),
            '--start_index', str(args.start_index)
        ]
        
        log_file = log_dir / f"worker_{i}_gpu_{phys_gpu}.log"
        with open(log_file, 'w') as f:
            processes.append((i, phys_gpu, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env), log_file))
        time.sleep(0.5)
    
    failed = 0
    for i, gpu, p, log in processes:
        p.wait()
        if p.returncode != 0:
            print(f"Worker {i} (GPU {gpu}) FAILED. Log: {log}")
            failed += 1
            
    if failed == 0:
        print(f"All success. Results: {args.data_root / args.out_obj_dir}")
    else:
        print(f"Completed with {failed} failures.")

if __name__ == "__main__":
    main()