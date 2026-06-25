import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from peft import LoraConfig
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class DiffGANGenerator(nn.Module):
    """SDXL-Inpainting UNet wrapper with LoRA injection and a trainable terminal conv_out."""
    def __init__(
        self,
        base_model_path: str,
        lora_rank: int = 256,
        lora_modules: List[str] = ["to_k", "to_q", "to_v", "to_out.0"],
        weight_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        logger.info(f"Loading SDXL UNet from {base_model_path}")
        
        # 1. Load UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model_path,
            subfolder="unet",
            torch_dtype=weight_dtype
        )
        
        # 2. Inject LoRA
        self._inject_lora(lora_rank, lora_modules)

        # 3. Critical: Unfreeze Output Layer for Full Fine-Tuning
        self._unfreeze_output_layer()
        
        self.weight_dtype = weight_dtype

    def _unfreeze_output_layer(self):
        """
        Explicitly enables gradient calculation for conv_out.
        This is necessary because PEFT/LoRA usually freezes the entire base model.
        """
        for param in self.unet.conv_out.parameters():
            param.requires_grad = True

    def _inject_lora(self, rank, target_modules):
        """Injects LoRA adapters"""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        self.unet.add_adapter(lora_config)
        logger.info(f"Injected LoRA with rank {rank}.")

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()

    def forward(
        self,
        z_in: torch.Tensor,
        timesteps: torch.Tensor,
        added_cond_kwargs: Dict,
        encoder_hidden_states: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        Args:
            z_in: Concatenated latents [Batch, 9, H, W]
            timesteps: [Batch]
            added_cond_kwargs: SDXL conditioning
            encoder_hidden_states: Text embeddings [Batch, Seq, Dim]
        """
        # Forward through UNet
        output = self.unet(
            z_in,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]
        
        return output