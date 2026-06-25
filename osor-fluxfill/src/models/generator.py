import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import logging
from peft import LoraConfig, get_peft_model
from diffusers import FluxTransformer2DModel

logger = logging.getLogger(__name__)

class FluxGenerator(nn.Module):
    """
    Wrapper around Flux DiT (FluxTransformer2DModel).
    
    Adaptations:
    1. Wraps FluxTransformer2DModel for Flow Matching.
    2. Supports LoRA injection (targeting Attention/FFN layers).
    3. Explicitly unfreezes the output projection (proj_out) for full fine-tuning.
    4. Provides utility methods for DiT-specific latent packing/unpacking.
    """
    def __init__(
        self,
        base_model_path: str,
        lora_rank: int = 16,
        lora_modules: List[str] = ["to_k", "to_q", "to_v", "to_out.0"], 
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        logger.info(f"Loading Flux Transformer from {base_model_path}")
        
        # 1. Load Transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype
        )
        
        self.weight_dtype = weight_dtype

        # 2. Inject LoRA
        self._inject_lora(lora_rank, lora_modules)
        
        # 3. Unfreeze Output Projection (Full Fine-tuning)
        # As seen in wCA, this helps the model adapt the final head better
        self._unfreeze_output_layer()

    def _inject_lora(self, rank, target_modules):
        """Injects LoRA adapters using PEFT"""
        logger.info(f"Injecting LoRA (Rank {rank}) into modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            bias="none",
        )
        # Wrap the transformer
        self.transformer = get_peft_model(self.transformer, lora_config)
        self.transformer.print_trainable_parameters()

    def _unfreeze_output_layer(self):
        """
        Explicitly enables gradient calculation for the output projection.
        This layer is crucial for the final mapping from DiT latent space to VAE space.
        """
        logger.info("Unfreezing Output Projection (proj_out) for full fine-tuning.")
        # In PEFT-wrapped models, the base model is under self.transformer.base_model
        target = self.transformer.base_model.model if hasattr(self.transformer, "base_model") else self.transformer
        
        if hasattr(target, "proj_out"):
            for param in target.proj_out.parameters():
                param.requires_grad = True
        else:
            logger.warning("Could not find 'proj_out' in Transformer model to unfreeze.")

    def enable_gradient_checkpointing(self):
        # Access the base transformer if wrapped in PEFT
        inner_model = self.transformer.base_model.model if hasattr(self.transformer, "base_model") else self.transformer
        inner_model.enable_gradient_checkpointing()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor = None,
        **kwargs
    ):
        """
        Forward pass for Flux DiT.
        
        Args:
            hidden_states: [B, L, C] - Packed and concatenated latents.
            timestep: [B]
            ...
        """
        # Call Transformer
        # PEFT model forward handles the wrapper logic
        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False
        )[0]
        
        return output

    # --- Static Utility Methods (Consistent with FluxFillPipeline) ---

    @staticmethod
    def pack_latents(latents, batch_size, num_channels_latents, height, width):
        """
        Packs 2D latents into 1D sequence using 2x2 patching.
        Args:
            latents: [B, C, H, W]
        Returns:
            packed: [B, (H/2)*(W/2), C*4]
        """
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def unpack_latents(latents, height, width):
        """
        Unpacks 1D sequence back to 2D latents.
        Args:
            latents: [B, L, C*4]
        Returns:
            unpacked: [B, C, H, W]
        """
        batch_size, num_patches, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height, width)
        
        return latents
