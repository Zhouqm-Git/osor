<h1 align="center">OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal</h1>

<p align="center">
  <strong>ECCV 2026</strong>
</p>

<p align="center">
  Qinming Zhou<sup>1,2,*</sup>,
  Chenxi Sun<sup>1,3,*</sup>,
  Deyang Kong<sup>1,3</sup>,
  Junhao He<sup>1</sup>,
  Xiangheng Tang<sup>1,4</sup>,
  Peike Yu<sup>1,5</sup>,
  Haotian Wu<sup>1</sup>,
  Leilei Cao<sup>6</sup>,
  Linfeng Zhang<sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Jiao Tong University,
  <sup>2</sup>Tsinghua University,
  <sup>3</sup>University of Electronic Science and Technology of China,
  <sup>4</sup>Xidian University,
  <sup>5</sup>Tongji University,
  <sup>6</sup>Transsion
</p>

<p align="center">
  <a href="https://zhouqm-git.github.io/osor/"><img src="https://img.shields.io/badge/Project-Page-62B01E?style=flat&labelColor=555555" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2606.28094"><img src="https://img.shields.io/badge/arXiv-2606.28094-B31B1B?style=flat&labelColor=555555" alt="arXiv"></a>
  <a href="https://huggingface.co/QinmingZhou/OSOR"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-1F8ACB?style=flat&labelColor=555555" alt="Hugging Face Model"></a>
  <a href="https://modelscope.cn/datasets/ZhouqmR/CORNE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-CORNE-1F8ACB?style=flat&labelColor=555555" alt="CORNE Dataset"></a>
  <a href="https://huggingface.co/datasets/QinmingZhou/CORNE-Val"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Bench-CORNE--Val-2EA44F?style=flat&labelColor=555555" alt="CORNE-Val Benchmark"></a>
  <a href="https://huggingface.co/datasets/QinmingZhou/AnimeEraseBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Bench-AnimeEraseBench-2EA44F?style=flat&labelColor=555555" alt="AnimeEraseBench"></a>
  <a href="https://huggingface.co/datasets/QinmingZhou/TextEraseBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Bench-TextEraseBench-2EA44F?style=flat&labelColor=555555" alt="TextEraseBench"></a>
</p>

OSOR is a one-step diffusion framework for **effect-aware** and **mask-robust** object removal. It removes the target together with associated shadows, reflections, and residual traces in a single denoising step.

![OSOR teaser](assets/first.png)

## Abstract

Real-world object removal is challenging due to two key difficulties: the target object's non-local effects, such as shadows and reflections, which are difficult to model, and the fact that user-provided masks are often inaccurate or incomplete. With billions of parameters and tens of denoising steps, diffusion-based models achieve strong removal performance at the expense of substantial computational cost, limiting their use in interactive applications and on edge devices. To address these challenges, we present **OSOR** (One-Step Object Removal), which simultaneously achieves efficient, effect-aware, and mask-robust object removal. Concretely, OSOR introduces: (1) an occupancy-guided discriminator for precise boundary supervision, enabling stable single-step diffusion training; (2) an alpha head that leverages knowledge from pretrained diffusion models to predict appropriate removal regions with minimal overhead, thereby handling imperfect masks; and (3) a semantic-anchored verification pipeline (SAVP) that filters noisy instruction-based triplets to produce effect-aware supervision at scale. Using SAVP, we curate **CORNE**, which contains 280K verified removal pairs, and further annotate AnimeEraseBench and TextEraseBench to evaluate performance on more complex removal tasks. Experiments show that OSOR surpasses strong multi-step diffusion baselines in perceptual quality while achieving 4x to 30x faster inference.

## Highlights

- **One-step inference.** OSOR processes a 1024 x 1024 image in under one second on a single A100 GPU and is 4x to 30x faster than multi-step diffusion baselines.
- **Effect-aware removal.** An alpha head learns an appropriate editing region beyond a conservative object mask, allowing OSOR to remove non-local effects while preserving the surrounding scene.
- **Comprehensive release.** We release two LoRA-based implementations, the SAVP data pipeline, 287,012 CORNE training pairs, and three paired-background benchmarks.

## News

- **2026.07**: We provide reproducible environment setup, model and dataset cards, and component-specific licensing information.
- **2026.06**: The [paper](https://arxiv.org/abs/2606.28094), code, model weights, CORNE, and three benchmarks are released.

## Released Resources

| Resource | Contents | Link |
| --- | --- | --- |
| Paper | ECCV 2026 paper and supplementary material | [arXiv:2606.28094](https://arxiv.org/abs/2606.28094) |
| Model weights | OSOR-FLUX-Fill and OSOR-SDXL-Inpainting, Phase I and Phase II | [QinmingZhou/OSOR](https://huggingface.co/QinmingZhou/OSOR) |
| CORNE | 287,012 effect-aware training pairs constructed from NHR-Edit with SAVP | [ModelScope](https://modelscope.cn/datasets/ZhouqmR/CORNE) |
| CORNE-Val | 219 held-out paired-background samples | [Hugging Face](https://huggingface.co/datasets/QinmingZhou/CORNE-Val) |
| AnimeEraseBench | 157 paired samples for anime object removal | [Hugging Face](https://huggingface.co/datasets/QinmingZhou/AnimeEraseBench) |
| TextEraseBench | 185 paired samples for scene-text removal | [Hugging Face](https://huggingface.co/datasets/QinmingZhou/TextEraseBench) |

## Repository Structure

| Directory | Contents |
| --- | --- |
| [`osor-fluxfill/`](osor-fluxfill) | FLUX.1 Fill [dev] implementation, training, and inference |
| [`osor-sdxlinpainting/`](osor-sdxlinpainting) | SDXL-Inpainting implementation, training, and inference |
| [`SAVP/`](SAVP) | Semantic-Anchored Verification Pipeline used to construct CORNE |
| [`docs/`](docs) | Source for the interactive [project page](https://zhouqm-git.github.io/osor/) |

## Quick Start

### Installation

The OSOR model code was tested with Python 3.10, PyTorch 2.6.0, and CUDA 12.4.

```bash
git clone https://github.com/Zhouqm-Git/osor.git
cd osor

conda create -n osor python=3.10 -y
conda activate osor

pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

For another CUDA version, install the corresponding PyTorch wheel before installing `requirements.txt`. SAVP uses a separate MMDetection and SAM2 environment; see [SAVP/README.md](SAVP/README.md).

### Download Checkpoints

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) and download either or both checkpoint families from [QinmingZhou/OSOR](https://huggingface.co/QinmingZhou/OSOR):

```bash
hf download QinmingZhou/OSOR \
  --include "osor-fluxfill/weights/*.pth" \
  --local-dir .

hf download QinmingZhou/OSOR \
  --include "osor-sdxlinpainting/weights/*.pth" \
  --local-dir .
```

Phase II is the effect-aware model used for the main paper setting. Phase I checkpoints are also provided for reproducing the two-stage training procedure.

### Prepare Inputs

For batch inference, place input images and binary masks in two directories. Image-mask pairs must share the same filename stem.

```text
inputs/
├── imgs/
│   ├── example_1.png
│   └── example_2.jpg
└── masks/
    ├── example_1.png
    └── example_2.png
```

### OSOR-FLUX-Fill

Set `model.flux_base` in `osor-fluxfill/configs/phase2.yaml` to the local [FLUX.1 Fill [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) directory, then cache the fixed prompt and run inference:

```bash
cd osor-fluxfill

python scripts/cache_prompts.py --config configs/phase2.yaml

python scripts/inference_enhance.py \
  -b /path/to/flux-fill-dev \
  -w weights/fluxfill_phase2.pth \
  -p pretrained_weights/fixed_prompt_embeds.pt \
  -i inputs/imgs \
  -m inputs/masks \
  -o outputs
```

### OSOR-SDXL-Inpainting

Set `model.sdxl_base` in `osor-sdxlinpainting/configs/phase2.yaml` to the local [SDXL-Inpainting 0.1](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) directory, then run:

```bash
cd osor-sdxlinpainting

python scripts/cache_prompts.py --config configs/phase2.yaml

python scripts/inference_enhance.py \
  -b /path/to/sdxl-inpainting \
  -w weights/sdxlinpainting_phase2.pth \
  -p pretrained_weights/fixed_prompt_embeds.pt \
  -i inputs/imgs \
  -m inputs/masks \
  -o outputs \
  --resolution 512
```

The paper setting uses the direct OSOR output. Paste-back post-processing is disabled by default because it can preserve unwanted effects outside the input mask; enable `--paste_back` only when compositing onto the original image is specifically desired.

## Training

Both implementations follow the same curriculum:

1. **Phase I:** one-step removal with hard latent blending and occupancy-guided discriminator supervision.
2. **Phase II:** alpha-aware adaptive blending with incomplete-mask conditioning.

After updating the model and dataset paths in the corresponding YAML files:

```bash
cd osor-fluxfill  # or osor-sdxlinpainting
bash scripts/train.sh 1
bash scripts/train.sh 2
```

See the [FLUX-Fill instructions](osor-fluxfill/README.md) and [SDXL-Inpainting instructions](osor-sdxlinpainting/README.md) for checkpoint layout and backbone-specific details.

## Building CORNE with SAVP

SAVP converts noisy instruction-based editing triplets into reliable object-removal supervision. It orders addition/removal pairs, verifies semantic changes with GroundingDINO, constructs object-core masks with SAM2, and fuses object and residual-effect regions into effect-aware masks.

The released pipeline includes NHR-Edit preparation, multi-GPU processing, held-out shard handling, and object-core extraction. See [SAVP/README.md](SAVP/README.md) for the complete reproduction workflow.

## License

The code, model weights, and datasets do not all share the same license:

| Component | License |
| --- | --- |
| Source code in this repository | MIT |
| OSOR-FLUX-Fill checkpoints | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/LICENSE.md) |
| OSOR-SDXL-Inpainting checkpoints | [CreativeML Open RAIL++-M](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1/blob/main/LICENSE.md) |
| CORNE and CORNE-Val | Apache License 2.0; derived from [NHR-Edit](https://huggingface.co/datasets/iitolstykh/NHR-Edit) |
| AnimeEraseBench and TextEraseBench | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The MIT license for the source code does not override the licenses of the base models, released checkpoints, or datasets. Please review the applicable terms before use or redistribution.

## Citation

If you find this repository, model, or dataset useful, please cite:

```bibtex
@inproceedings{zhou2026osor,
  title     = {OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal},
  author    = {Zhou, Qinming and Sun, Chenxi and Kong, Deyang and He, Junhao and Tang, Xiangheng and Yu, Peike and Wu, Haotian and Cao, Leilei and Zhang, Linfeng},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.28094}
}
```

CORNE and CORNE-Val are derived from NHR-Edit. Please also cite [NoHumansRequired](https://arxiv.org/abs/2507.14119) when using either dataset.

## Acknowledgements

OSOR builds on [FLUX.1 Fill [dev]](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), [SDXL-Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), and [OpenCLIP](https://github.com/mlfoundations/open_clip). SAVP uses the [MMDetection MM-Grounding-DINO implementation](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) and [SAM2](https://github.com/facebookresearch/sam2), and CORNE is constructed from [NHR-Edit](https://huggingface.co/datasets/iitolstykh/NHR-Edit). We thank the authors for making their work available.
