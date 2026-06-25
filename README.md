# OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal

OSOR is a one-step diffusion framework for efficient, effect-aware, and mask-robust object removal. It removes both the target object and object-associated effects such as shadows, reflections, and residual traces, while running in a single denoising step.

![OSOR teaser](assets/first.png)

## Abstract

Real-world object removal is challenging due to two key difficulties: the target object's non-local effects, such as shadows and reflections, which are difficult to model, and the fact that user-provided masks are often inaccurate or incomplete. With billions of parameters and tens of denoising steps, diffusion-based models achieve strong removal performance at the expense of substantial computational cost, limiting their use in interactive applications and on edge devices. To address these challenges, we present **OSOR** (One-Step Object Removal), which simultaneously achieves efficient, effect-aware, and mask-robust object removal. Concretely, OSOR introduces: (1) an occupancy-guided discriminator for precise boundary supervision, enabling stable single-step diffusion training; (2) an alpha head that leverages knowledge from pretrained diffusion models to predict appropriate removal regions with minimal overhead, thereby handling imperfect masks; and (3) a semantic-anchored verification pipeline (SAVP) that filters noisy instruction-based triplets to produce effect-aware supervision at scale. Using SAVP, we curate **CORNE**, which contains 280K verified removal pairs, and further annotate AnimeEraseBench and TextEraseBench to evaluate performance on more complex removal tasks. Experiments show that OSOR surpasses strong multi-step diffusion baselines in perceptual quality while achieving 4x to 30x faster inference. Datasets, models, and code will be released.

## Release

- **SAVP**: semantic-anchored verification pipeline for building CORNE-style effect-aware masks.
- **OSOR-FLUX-Fill**: training and inference code for the FLUX-Fill backbone.
- **OSOR-SDXL-Inpainting**: training and inference code for the SDXL-Inpainting backbone.
- **Checkpoints and datasets**: coming soon.

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

## Citation

```bibtex
@article{zhou2026osor,
  title   = {OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal},
  author  = {Zhou, Qinming and Sun, Chenxi and Kong, Deyang and He, Junhao and Tang, Xiangheng and Yu, Peike and Wu, Haotian and Cao, Leilei and Zhang, Linfeng},
  year    = {2026}
}
```

## Acknowledgements

This project builds on open-source diffusion, inpainting, segmentation, and detection toolchains. We thank the authors and maintainers of these projects.
