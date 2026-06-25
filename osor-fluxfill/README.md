# OSOR-FLUX-Fill

This package contains the FLUX-Fill implementation of OSOR.

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

Edit `configs/phase1.yaml` and `configs/phase2.yaml` to point to your local FLUX-Fill base model, OpenCLIP checkpoint, and CORNE-style data lists.

```bash
cd osor-fluxfill
bash scripts/train.sh 1
bash scripts/train.sh 2
```

## Checkpoints

Released OSOR-FLUX-Fill checkpoints are hosted at [QinmingZhou/OSOR](https://huggingface.co/QinmingZhou/OSOR):

```bash
cd ..
hf download QinmingZhou/OSOR \
  --include "osor-fluxfill/weights/*.pth" \
  --local-dir .
cd osor-fluxfill
```

This creates:

```text
weights/fluxfill_phase1.pth
weights/fluxfill_phase2.pth
```

The fixed prompt embedding file is generated automatically by `scripts/train.sh` if missing. For inference, generate it once with:

```bash
python scripts/cache_prompts.py --config configs/phase2.yaml
```

Phase II inference:

```bash
python scripts/inference_enhance.py \
  -b /path/to/flux-fill-dev \
  -w weights/fluxfill_phase2.pth \
  -p pretrained_weights/fixed_prompt_embeds.pt \
  -i inputs/imgs \
  -m inputs/masks \
  -o outputs \
  --paste_back
```

See the root README for the full OSOR release overview.
