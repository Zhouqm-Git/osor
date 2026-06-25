import argparse
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Processing Launcher")
    parser.add_argument('--bg-dir', required=True, type=Path, help='Background images directory')
    parser.add_argument('--shot-dir', required=True, type=Path, help='Shot images directory')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prompt-file', required=True, type=Path, help='JSON file with prompts')
    
    parser.add_argument('--gpus', required=True, type=str, 
                       help='Comma-separated GPU IDs, e.g., "0,1,2,3"')
    parser.add_argument('--workers-per-gpu', type=int, default=6,
                       help='Number of worker processes per GPU (default: 6)')
    parser.add_argument('--log-interval', type=int, default=100, 
                       help='Log progress every N samples')
    
    parser.add_argument('--dino-config', type=str)
    parser.add_argument('--dino-checkpoint', type=str)
    parser.add_argument('--sam-config', type=str)
    parser.add_argument('--sam-checkpoint', type=str)
    
    args = parser.parse_args()
    
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    total_workers = num_gpus * args.workers_per_gpu
    
    print(f"Launching {total_workers} workers on {num_gpus} GPUs ({args.workers_per_gpu} workers per GPU)")
    print(f"GPUs: {gpu_ids}")
    
    processes = []
    for worker_index in range(total_workers):
        physical_gpu_id = gpu_ids[worker_index // args.workers_per_gpu]
        
        cmd = [
            'python', 'scripts/gpu_worker.py',
            '--bg-dir', str(args.bg_dir),
            '--shot-dir', str(args.shot_dir),
            '--output-dir', str(args.output_dir),
            '--prompt-file', str(args.prompt_file),
            '--gpu-id', str(physical_gpu_id),
            '--total-workers', str(total_workers),
            '--worker-index', str(worker_index),
            '--log-interval', str(args.log_interval)
        ]
        if args.dino_config:
            cmd.extend(['--dino-config', args.dino_config])
        if args.dino_checkpoint:
            cmd.extend(['--dino-checkpoint', args.dino_checkpoint])
        if args.sam_config:
            cmd.extend(['--sam-config', args.sam_config])
        if args.sam_checkpoint:
            cmd.extend(['--sam-checkpoint', args.sam_checkpoint])
        
        log_file = args.output_dir / f"worker_{worker_index}_gpu_{physical_gpu_id}_stdout.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Starting Worker {worker_index} -> GPU {physical_gpu_id}...")
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            processes.append((worker_index, physical_gpu_id, proc, log_file))
        
        time.sleep(2)
    
    print(f"\nAll {total_workers} workers launched. Waiting for completion...\n")
    
    for worker_index, gpu_id, proc, log_file in processes:
        proc.wait()
        print(f"Worker {worker_index} (GPU {gpu_id}) finished (exit code: {proc.returncode})")
        print(f"  Log: {log_file}")
    
    print("\nAll processes completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

