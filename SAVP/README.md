# SAVP: Semantic-Anchored Verification Pipeline

SAVP is the data verification and mask synthesis pipeline used by OSOR to build effect-aware object-removal supervision from noisy instruction-based image-editing triplets.

Given an original image, an edited image, and an edit instruction, SAVP:

1. orders the pair as `(I_shot, I_gt)` according to whether the instruction is an add or remove edit;
2. computes a robust multi-feature difference mask from log-luminance, chromaticity, and gradient differences;
3. verifies that the localized difference is semantically aligned with GroundingDINO boxes from the instruction;
4. uses SAM2 to obtain an object-core mask from validated boxes;
5. fuses the validated difference mask and object-core mask to produce an effect-aware target mask.

This corresponds to the "Semantic-anchored verification" and "Effect-aware mask synthesis" sections in the OSOR paper and supplementary material.

## Repository Layout

```text
SAVP/
├── core/
│   ├── config.py              # Reproduction constants and model paths
│   ├── difference.py          # Multi-feature perceptual difference masks
│   ├── geometry_validator.py  # Semantic/difference box validation
│   ├── grounding_dino.py      # MM Grounding DINO wrapper
│   ├── pipeline.py            # SAVP / EditMaskPipeline orchestrator
│   └── sam_segmenter.py       # SAM2 wrapper
├── scripts/
│   ├── prepare_dataset.py                 # Prepare one NHR-Edit shard
│   ├── prepare_all_shards.py              # Prepare all add/remove shards
│   ├── run_multi_gpu.py                   # Multi-GPU SAVP launcher
│   ├── gpu_worker.py                      # Per-worker processing
│   ├── run_separate_shadows_multi_gpu.py  # Object-core extraction launcher
│   └── separate_shadows.py                # Effect-heavy sample post-process
└── README.md
```

Debug-only utilities, shell wrappers, internal visualization scripts, and cluster-specific launchers are intentionally not included in the open-source version.

## Installation

Create an environment:

```bash
conda create -n savp python=3.10 -y
conda activate savp

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2.0" opencv-python scikit-image scipy matplotlib datasets tqdm pyarrow
```

Install MMDetection for GroundingDINO:

```bash
cd /path/to/SAVP
git clone https://github.com/open-mmlab/mmdetection.git
pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
cd mmdetection
pip install -v -e . --no-build-isolation
cd ..
pip install transformers==4.29.2
```

Install SAM2:

```bash
cd /path/to/SAVP
git clone https://github.com/facebookresearch/segment-anything-2.git sam2
cd sam2
sed -i 's/torch>=2.5.1/torch>=2.4.0/g' setup.py
sed -i 's/torchvision>=0.20.1/torchvision>=0.19.0/g' setup.py
pip install -e .
cd ..
```

Place checkpoints in the default locations, or pass overrides to the scripts:

```text
SAVP/
├── checkpoints/
│   └── grounding_dino_swin-l_pretrain_all-56d69e78.pth
└── sam2/
    └── checkpoints/
        └── sam2.1_hiera_large.pt
```

The default paths are defined in `core/config.py`. You can override them with:

```bash
--dino-config /path/to/grounding_dino_config.py
--dino-checkpoint /path/to/grounding_dino_checkpoint.pth
--sam-config configs/sam2.1/sam2.1_hiera_l.yaml
--sam-checkpoint /path/to/sam2.1_hiera_large.pt
```

## Prepare NHR-Edit Pairs

Download NHR-Edit parquet shards first. For example:

```bash
hf download iitolstykh/NHR-Edit --type dataset --local-dir /path/to/NHR-Edit
```

Prepare one shard:

```bash
cd /path/to/SAVP
python scripts/prepare_dataset.py \
  --data-dir /path/to/NHR-Edit \
  --output-root demo \
  --shard-id 0 \
  --num-shards 37 \
  --num-workers 32
```

Prepare all shards:

```bash
python scripts/prepare_all_shards.py \
  --data-dir /path/to/NHR-Edit \
  --output-root /path/to/corne_workdir \
  --num-shards 37 \
  --num-workers 32
```

Following the paper setup, `prepare_all_shards.py` skips shards `10`, `24`, `30`, and `31`, which are reserved for held-out sampling and CORNE-Val construction.

The preparation scripts keep the paper's ordered-pair logic:

```text
Add object:
  source_image -> bg / I_gt
  edited_image -> shot / I_shot

Remove object:
  edited_image -> bg / I_gt
  source_image -> shot / I_shot
```

The resulting folder has:

```text
demo/
├── bg/
├── shot/
└── prompts.json
```

`prompts.json` maps each image stem to its edit instruction:

```json
{
  "000000": "Add sport sunglasses to the man's face.",
  "000001": "Remove the backpack from the person."
}
```

## Run SAVP

Run the multi-GPU pipeline:

```bash
python scripts/run_multi_gpu.py \
  --bg-dir demo/bg \
  --shot-dir demo/shot \
  --output-dir demo/output \
  --prompt-file demo/prompts.json \
  --gpus "0,1,2,3" \
  --workers-per-gpu 6 \
  --log-interval 100
```

The main output is:

```text
demo/output/
├── masks/                         # effect-aware masks m_gt
├── worker_0_gpu_0_log.txt
└── worker_0_gpu_0_stdout.log
```

By default `gpu_worker.py` writes masks as `masks/mask_<stem>.png`.

## Extract Object-Core Masks

For Phase II incomplete-mask conditioning, the paper uses tight object-core masks and effect-heavy sample selection. After generating effect-aware masks, run:

```bash
python scripts/run_separate_shadows_multi_gpu.py \
  --data-root demo \
  --img-dir shot \
  --mask-dir output/masks \
  --out-obj-dir mask_sam \
  --shadow-thresh 0.25 \
  --gpus "0,1,2,3" \
  --workers-per-gpu 2
```

This writes object-core masks to:

```text
demo/mask_sam/
```

`separate_shadows.py` accepts masks named either `<stem>.png` or `mask_<stem>.png`.

## Reproduction Constants

The open-source code keeps the SAVP constants reported in the supplementary material:

| Symbol | Value | Implementation |
| --- | ---: | --- |
| `w_L` | 0.6 | `DifferenceConfig.w_l` |
| `w_ab` | 0.3 | `DifferenceConfig.w_ab` |
| `w_T` | 0.1 | `DifferenceConfig.w_tex` |
| `tau_H` | 0.07 | `DifferenceConfig.threshold` |
| `A_min` | 2000 | `DifferenceConfig.min_area` |
| `A_hole` | 500 | `DifferenceConfig.max_hole_area` |
| `alpha` | 0.3 | `DifferenceConfig.area_threshold_ratio` |
| `N_max` | 4 | `DifferenceConfig.max_main_regions` |
| `tau_noise` | 0.3 | `DifferenceConfig.noise_threshold` |
| `tau_score` | 0.3 | `GroundingDINOConfig.score_threshold` |
| `K_sem` | 3 | `GroundingDINOConfig.top_k` |
| `tau_iou` | 0.3 | `GeometryConfig.iou_threshold` |
| `tau_scale` | 2.0 | `GeometryConfig.scale_threshold` |
| `tau_keep` | 0.5 | connected-component keep ratio in `pipeline.py` |
| `r_dilate` | 1.2 | `MaskConfig.expand_ratio` |
| `tau_eff` | 0.25 | `--shadow-thresh 0.25` for object-core extraction |

## Script Selection

The open-source version keeps scripts that are needed to reproduce the data construction pipeline:

```text
prepare_dataset.py
prepare_all_shards.py
run_multi_gpu.py
gpu_worker.py
run_separate_shadows_multi_gpu.py
separate_shadows.py
```

Shell wrappers, debug scripts, cluster-specific `sbatch` launchers, and internal figure/ranking utilities were removed to avoid stale machine paths and non-essential dependencies.
