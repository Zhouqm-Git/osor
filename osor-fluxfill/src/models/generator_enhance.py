import torch
import torch.nn as nn
from .generator import FluxGenerator

class FluxGeneratorEnhance(FluxGenerator):
    def __init__(
        self,
        base_model_path: str,
        lora_rank: int = 16,
        lora_modules: list = ["to_k", "to_q", "to_v", "to_out.0"],
        weight_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(
            base_model_path=base_model_path,
            lora_rank=lora_rank,
            lora_modules=lora_modules,
            weight_dtype=weight_dtype
        )
        self._modify_output_projection()
        self._unfreeze_output_layer()

    def _modify_output_projection(self):
        if hasattr(self.transformer, "base_model"):
            target_model = self.transformer.base_model.model
        else:
            target_model = self.transformer

        old_proj = target_model.proj_out
        if old_proj.out_features == 68:
            return

        new_proj = nn.Linear(
            in_features=old_proj.in_features,
            out_features=68,
            bias=old_proj.bias is not None,
            dtype=old_proj.weight.dtype,
            device=old_proj.weight.device
        )

        with torch.no_grad():
            new_proj.weight.data[:64, :] = old_proj.weight.data
            if old_proj.bias is not None:
                new_proj.bias.data[:64] = old_proj.bias.data

            torch.nn.init.kaiming_normal_(new_proj.weight.data[64:, :], nonlinearity='relu')
            
            if new_proj.bias is not None:
                new_proj.bias.data[64:].zero_()
        
        target_model.proj_out = new_proj

    def forward(self, hidden_states, **kwargs):
        output = super().forward(hidden_states=hidden_states, **kwargs)
        v_pred = output[..., :64]
        alpha_logits = output[..., 64:]
        return v_pred, alpha_logits