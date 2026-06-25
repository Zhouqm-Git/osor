# OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal

OSOR is a one-step diffusion framework for efficient, effect-aware, and mask-robust object removal. It removes both the target object and object-associated effects such as shadows, reflections, and residual traces, while running in a single denoising step.

![OSOR teaser](assets/first.png)

## Abstract

Real-world object removal is challenging due to two key difficulties: the target object's non-local effects, such as shadows and reflections, which are difficult to model, and the fact that user-provided masks are often inaccurate or incomplete. With billions of parameters and tens of denoising steps, diffusion-based models achieve strong removal performance at the expense of substantial computational cost, limiting their use in interactive applications and on edge devices. To address these challenges, we present **OSOR** (One-Step Object Removal), which simultaneously achieves efficient, effect-aware, and mask-robust object removal. Concretely, OSOR introduces: (1) an occupancy-guided discriminator for precise boundary supervision, enabling stable single-step diffusion training; (2) an alpha head that leverages knowledge from pretrained diffusion models to predict appropriate removal regions with minimal overhead, thereby handling imperfect masks; and (3) a semantic-anchored verification pipeline (SAVP) that filters noisy instruction-based triplets to produce effect-aware supervision at scale. Using SAVP, we curate **CORNE**, which contains 280K verified removal pairs, and further annotate AnimeEraseBench and TextEraseBench to evaluate performance on more complex removal tasks. Experiments show that OSOR surpasses strong multi-step diffusion baselines in perceptual quality while achieving 4x to 30x faster inference.

## Release

- **Code**: SAVP, OSOR-FLUX-Fill, and OSOR-SDXL-Inpainting are released in this repository.
- **Training data**: [CORNE](https://modelscope.cn/datasets/ZhouqmR/CORNE) on ModelScope.
- **Benchmarks**: [CORNE-Val](https://huggingface.co/datasets/QinmingZhou/CORNE-Val), [AnimeEraseBench](https://huggingface.co/datasets/QinmingZhou/AnimeEraseBench), and [TextEraseBench](https://huggingface.co/datasets/QinmingZhou/TextEraseBench) on Hugging Face.
- **Model weights**: [QinmingZhou/OSOR](https://huggingface.co/QinmingZhou/OSOR) on Hugging Face.

## Repository Layout

```text
.
├── SAVP/                    # Semantic-Anchored Verification Pipeline
├── osor-fluxfill/           # OSOR with the FLUX-Fill backbone
├── osor-sdxlinpainting/     # OSOR with the SDXL-Inpainting backbone
└── assets/
    └── first.png
```

## SAVP Data Construction

SAVP extracts reliable object-removal supervision from noisy instruction-based editing triplets. It verifies localized semantic changes with GroundingDINO, synthesizes effect-aware masks with perceptual difference analysis and SAM2, and derives object-core masks for incomplete-mask conditioning.

See [SAVP/README.md](SAVP/README.md) for installation, NHR-Edit preparation, multi-GPU processing, object-core extraction, and reproduction constants.

## OSOR Training And Inference

Both backbone releases follow the same two-phase curriculum:

1. **Phase I** trains one-step removal with hard latent blending and occupancy-guided discriminator supervision.
2. **Phase II** adds alpha prediction and trains with incomplete-mask conditioning for robust effect removal.

Backbone-specific instructions:

- [osor-fluxfill](osor-fluxfill): FLUX-Fill implementation.
- [osor-sdxlinpainting](osor-sdxlinpainting): SDXL-Inpainting implementation.

Each package contains `configs/`, `scripts/`, and `src/` and can be used independently.

Download released checkpoints with:

```bash
hf download QinmingZhou/OSOR \
  --include "osor-fluxfill/weights/*.pth" \
  --local-dir .

hf download QinmingZhou/OSOR \
  --include "osor-sdxlinpainting/weights/*.pth" \
  --local-dir .
```

The released checkpoint files are:

```text
osor-fluxfill/weights/fluxfill_phase1.pth
osor-fluxfill/weights/fluxfill_phase2.pth
osor-sdxlinpainting/weights/sdxlinpainting_phase1.pth
osor-sdxlinpainting/weights/sdxlinpainting_phase2.pth
```

## Acknowledgements

This project builds on open-source diffusion, inpainting, segmentation, and detection toolchains. We thank the authors and maintainers of these projects.
