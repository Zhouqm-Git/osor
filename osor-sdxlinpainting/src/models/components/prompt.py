import torch
import logging

logger = logging.getLogger(__name__)

class SDXLFixedPromptEmbedder:
    """
    Loads pre-computed fixed prompt embeddings for SDXL.
    Replaces the heavy CLIP Text Encoders during training.
    """
    def __init__(self, cache_path: str, device: torch.device, dtype: torch.dtype = torch.float32):
        logger.info(f"Loading fixed prompt embeddings from {cache_path}")
        try:
            data = torch.load(cache_path, map_location="cpu")
        except FileNotFoundError:
            raise FileNotFoundError(f"Embedding file not found at {cache_path}. Please run 'scripts/cache_prompts.py' first.")
            
        self.prompt_embeds = data["prompt_embeds"].to(device=device, dtype=dtype)             # [1, 77, 2048]
        self.pooled_prompt_embeds = data["pooled_prompt_embeds"].to(device=device, dtype=dtype) # [1, 1280]
        
        logger.info(f"Loaded embeddings for prompt: '{data.get('prompt', 'Unknown')}'")

    def get_embeddings(self, batch_size: int):
        """
        Returns:
            prompt_embeds: [B, 77, 2048]
            pooled_prompt_embeds: [B, 1280]
        """
        # Efficient expansion without memory duplication
        prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
        pooled = self.pooled_prompt_embeds.expand(batch_size, -1)
        return prompt_embeds, pooled