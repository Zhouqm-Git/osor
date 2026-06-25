import argparse
import torch
import os
from omegaconf import OmegaConf
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

def cache_prompts(config_path):
    conf = OmegaConf.load(config_path)
    
    base_path = conf.model.sdxl_base
    output_path = conf.model.prompt_embeds_path
    fixed_prompt = conf.model.fixed_prompt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"--- SDXL Fixed Prompt Caching ---")
    print(f"Model Path: {base_path}")
    print(f"Prompt: '{fixed_prompt}'")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Loading Tokenizers and Encoders (this may take a moment)...")
    
    # SDXL Text Encoder 1
    tokenizer_1 = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer")
    text_encoder_1 = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
    
    # SDXL Text Encoder 2
    tokenizer_2 = CLIPTokenizer.from_pretrained(base_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16).to(device)

    print("Encoding...")
    with torch.no_grad():
        # Encoder 1
        text_inputs_1 = tokenizer_1(
            fixed_prompt, padding="max_length", max_length=tokenizer_1.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids_1 = text_inputs_1.input_ids.to(device)
        prompt_embeds_1 = text_encoder_1(text_input_ids_1, output_hidden_states=True).hidden_states[-2]

        # Encoder 2
        text_inputs_2 = tokenizer_2(
            fixed_prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(device)
        output_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
        prompt_embeds_2 = output_2.hidden_states[-2]
        pooled_prompt_embeds = output_2.text_embeds

        # Concatenate
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1) # [1, 77, 768+1280=2048]

    save_data = {
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
        "prompt": fixed_prompt
    }
    torch.save(save_data, output_path)
    print(f"Successfully saved embeddings to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()
    cache_prompts(args.config)