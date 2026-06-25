import argparse
import sys
import os
import logging
from omegaconf import OmegaConf
import torch

if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    _original_torch_load = torch.load
    
    def _force_device_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = f"cuda:{local_rank}"
        return _original_torch_load(*args, **kwargs)
    
    torch.load = _force_device_load

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.engine.train_phase1 import Phase1Trainer
from src.engine.train_phase2 import Phase2Trainer

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    phase = config.project.phase

    print(f"Initializing Training for Phase {phase}")

    if phase == 1:
        trainer = Phase1Trainer(config)
    elif phase == 2:
        trainer = Phase2Trainer(config)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    trainer.run()

if __name__ == "__main__":
    main()