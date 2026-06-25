from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.generator import DiffGANGenerator


class DiffGANGeneratorEnhance(DiffGANGenerator):
    """DiffGANGenerator with a 5-channel conv_out: 4 denoising channels + 1 alpha logit channel."""

    def __init__(
        self,
        base_model_path: str,
        lora_rank: int = 256,
        lora_modules=["to_k", "to_q", "to_v", "to_out.0"],
        weight_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(base_model_path, lora_rank, lora_modules, weight_dtype)
        # Replace the 4-channel conv_out with a 5-channel version (preserving the
        # pretrained denoising weights) and unfreeze it for joint optimization.
        self._modify_output_layer()
        self._unfreeze_output_layer()

    def _modify_output_layer(self):
        """Expand conv_out from 4 to 5 channels.

        Channels 0-3 are copied from the pretrained conv; channel 4 (alpha logit)
        is newly initialized with Kaiming-normal weights and zero bias.
        """
        old_conv = self.unet.conv_out

        if old_conv.out_channels == 5:
            return

        new_conv = nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=5,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight.data[:4] = old_conv.weight.data
            if old_conv.bias is not None:
                new_conv.bias.data[:4] = old_conv.bias.data

            torch.nn.init.kaiming_normal_(new_conv.weight.data[4:], nonlinearity='relu')

            if new_conv.bias is not None:
                new_conv.bias.data[4:].zero_()

        self.unet.conv_out = new_conv

    def forward(
        self,
        z_in: torch.Tensor,
        timesteps: torch.Tensor,
        added_cond_kwargs: Dict,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.unet(
            z_in,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]  # [B, 5, H, W]

        eps = output[:, :4, :, :]
        mask_logits = output[:, 4:, :, :]
        return eps, mask_logits
