"""Configuration for SAVP.

Default paths assume this open-source layout:

    SAVP/
    ├── checkpoints/
    ├── mmdetection/
    ├── sam2/
    └── ...

Paths can also be overridden from the command line in the provided scripts.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DifferenceConfig:
    threshold: float = 0.07
    min_area: int = 2000
    max_hole_area: int = 500
    w_l: float = 0.6
    w_ab: float = 0.3
    w_tex: float = 0.1
    noise_threshold: float = 0.3
    area_threshold_ratio: float = 0.3
    max_main_regions: int = 4


@dataclass
class GroundingDINOConfig:
    config_file: str = str(Path(__file__).parent.parent / "mmdetection/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py")
    checkpoint: str = str(Path(__file__).parent.parent / "checkpoints/grounding_dino_swin-l_pretrain_all-56d69e78.pth")
    device: str = "cuda:0"
    score_threshold: float = 0.3
    top_k: int = 3


@dataclass
class SAM2Config:
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint: str = str(Path(__file__).parent.parent / "sam2/checkpoints/sam2.1_hiera_large.pt")
    device: str = "cuda:0"


@dataclass
class GeometryConfig:
    iou_threshold: float = 0.3
    scale_threshold: float = 2


@dataclass
class MaskConfig:
    feather_px: int = 8
    expand_ratio: float = 1.2


@dataclass
class PipelineConfig:
    difference: DifferenceConfig = field(default_factory=DifferenceConfig)
    grounding_dino: GroundingDINOConfig = field(default_factory=GroundingDINOConfig)
    sam2: SAM2Config = field(default_factory=SAM2Config)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
