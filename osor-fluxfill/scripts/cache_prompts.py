import argparse
import torch
import os
from omegaconf import OmegaConf
from diffusers import FluxFillPipeline

def cache_prompts(config_path):
    conf = OmegaConf.load(config_path)
    
    # Read from config
    base_path = conf.model.flux_base
    output_path = conf.model.prompt_embeds_path
    fixed_prompt = conf.model.fixed_prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"--- Flux Fixed Prompt Caching ---")
    print(f"Model Path: {base_path}")
    print(f"Prompt: '{fixed_prompt}'")
    print(f"Output: {output_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading Flux Fill Pipeline (this may take a moment and requires significant VRAM)...")
    pipe = FluxFillPipeline.from_pretrained(
        base_path, 
        torch_dtype=dtype
    ).to(device)

    print("Encoding...")
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=fixed_prompt,
            prompt_2=fixed_prompt, # Use same prompt for T5
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512
        )

    if text_ids.dim() == 3:
        text_ids = text_ids.squeeze(0)

    save_data = {
        "prompt": fixed_prompt,
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
        "text_ids": text_ids.detach().cpu()  # Should be [Seq_Len, 3]
    }
    
    torch.save(save_data, output_path)
    print(f"Successfully saved embeddings to: {output_path}")
    print(f"Shapes: T5 {prompt_embeds.shape}, CLIP {pooled_prompt_embeds.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cache_prompts(args.config)
