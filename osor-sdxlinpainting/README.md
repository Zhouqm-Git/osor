# OSOR-SDXL-Inpainting

This package contains the SDXL-Inpainting implementation of OSOR.

## Structure

```text
configs/
├── phase1.yaml   # hard-blend one-step removal training
└── phase2.yaml   # alpha-aware incomplete-mask refinement
scripts/
├── train.py
├── train.sh
├── cache_prompts.py
├── inference.py
└── inference_enhance.py
src/
```

## Quick Start

Edit `configs/phase1.yaml` and `configs/phase2.yaml` to point to your local SDXL-Inpainting base model, OpenCLIP checkpoint, and CORNE-style data lists.

```bash
cd osor-sdxlinpainting
bash scripts/train.sh 1
bash scripts/train.sh 2
```

Phase II inference:

```bash
python scripts/inference_enhance.py \
  -b /path/to/sdxl-inpainting \
  -w /path/to/osor_sdxl_phase2.pth \
  -p pretrained_weights/fixed_prompt_embeds.pt \
  -i inputs/imgs \
  -m inputs/masks \
  -o outputs \
  --paste_back \
  --resolution 512
```

See the root README for the full OSOR release overview.
