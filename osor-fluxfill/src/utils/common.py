import importlib
import torch
import logging

logger = logging.getLogger(__name__)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class SuppressLogging:
    def __init__(self, level=logging.WARNING):
        self.level = level
        self.logger = logging.getLogger()
        self.original_level = self.logger.getEffectiveLevel()

    def __enter__(self):
        self.logger.setLevel(self.level)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)

def print_vram_state(prefix=None, logger=None):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        msg = f"VRAM: {allocated:.2f}GB / {reserved:.2f}GB"
        if prefix:
            msg = f"{prefix} - {msg}"
        if logger:
            logger.info(msg)
        else:
            print(msg)
        return allocated, reserved, torch.cuda.max_memory_allocated() / 1e9
    return 0, 0, 0
