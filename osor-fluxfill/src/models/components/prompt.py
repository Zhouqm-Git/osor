import os

import torch
import torch.nn as nn


class FluxFixedPromptEmbedder(nn.Module):
    """Loads pre-computed FLUX prompt embeddings (T5 + CLIP) from disk.

    Avoids loading the heavy text encoders during training.
    """

    def __init__(self, cache_path: str, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Prompt cache not found at {cache_path}")

        print(f"Loading fixed prompt embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")

        self.prompt_embeds = cache["prompt_embeds"].to(device=device, dtype=dtype)
        self.pooled_prompt_embeds = cache["pooled_prompt_embeds"].to(device=device, dtype=dtype)

        # FLUX transformer expects 2D txt_ids of shape [seq_len, 3].
        seq_len = self.prompt_embeds.shape[1]
        self.txt_ids = torch.zeros(seq_len, 3, device=device, dtype=dtype)

    def get_embeddings(self, batch_size):
        pe = self.prompt_embeds.repeat(batch_size, 1, 1)
        ppe = self.pooled_prompt_embeds.repeat(batch_size, 1)
        return pe, ppe, self.txt_ids
