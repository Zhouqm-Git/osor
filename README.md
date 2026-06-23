# OSOR: One-Step Diffusion Inpainting for Effect-Aware Object Removal

OSOR is a one-step diffusion object removal framework designed to be efficient, effect-aware, and robust to imperfect masks.

## Abstract

Real-world object removal is challenging due to two key difficulties: the target object's non-local effects, such as shadows and reflections, which are difficult to model, and the fact that user-provided masks are often inaccurate or incomplete. With billions of parameters and tens of denoising steps, diffusion-based models achieve strong removal performance at the expense of substantial computational cost, limiting their use in interactive applications and on edge devices. To address these challenges, we present **OSOR** (One-Step Object Removal), which simultaneously achieves efficient, effect-aware, and mask-robust object removal. Concretely, OSOR introduces: (1) an occupancy-guided discriminator for precise boundary supervision, enabling stable single-step diffusion training; (2) an alpha head that leverages knowledge from pretrained diffusion models to predict appropriate removal regions with minimal overhead, thereby handling imperfect masks; and (3) a semantic-anchored verification pipeline (SAVP) that filters noisy instruction-based triplets to produce effect-aware supervision at scale. Using SAVP, we curate **CORNE**, which contains 280K verified removal pairs, and further annotate AnimeEraseBench and TextEraseBench to evaluate performance on more complex removal tasks. Experiments show that OSOR surpasses strong multi-step diffusion baselines in perceptual quality while achieving 4x to 30x faster inference. Datasets, models, and code will be released.

## Teaser

![OSOR teaser](assets/first.png)

## Coming Soon

Code, checkpoints, datasets, and demos are coming soon.
