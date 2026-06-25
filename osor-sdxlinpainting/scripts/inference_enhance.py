#!/usr/bin/env python3
"""
DiffGAN-Removal Inference Script (Enhance / Phase II)
Adapts to Phase II training:
- Uses DiffGANGeneratorEnhance (5-channel conv_out)
- Uses predicted Alpha for blending
- Loads full fine-tuned weights (conv_out + LoRA)
- Uses Fixed Prompt Embeddings
"""

import os
import sys
import argparse
import glob
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Literal, List, Tuple, Optional
from torchvision.transforms.functional import to_tensor
from diffusers import AutoencoderKL, DDPMScheduler

# Add project root to path to allow imports from src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.generator_enhance import DiffGANGeneratorEnhance
# --- ADDED: Import Prompt Component ---
from src.models.components.prompt import SDXLFixedPromptEmbedder

class SDXLInferenceEnhance:
    """
    Self-contained SDXL inference class for object removal tasks (Phase II Enhanced).
    """
    
    def __init__(
        self,
        base_model_path: str,
        weight_path: str,
        prompt_embeds_path: str, # --- ADDED ---
        lora_modules: List[str],
        lora_rank: int,
        model_t: int,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.prompt_embeds_path = prompt_embeds_path # --- ADDED ---
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.device = device
        self.weight_dtype = weight_dtype

        # Initialize models
        self._init_vae()
        self._init_scheduler()
        self._init_generator()
        self._init_prompt_embedder() # --- ADDED ---
                
    def _init_vae(self):
        """Initialize VAE model"""
        print(f"Loading VAE from {self.base_model_path}...")
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, 
            subfolder="vae", 
            torch_dtype=self.weight_dtype
        ).to(self.device)
        self.vae.eval().requires_grad_(False)
        
    def _init_scheduler(self):
        """Initialize DDPM scheduler"""
        print(f"Loading Scheduler from {self.base_model_path}...")
        self.scheduler = DDPMScheduler.from_pretrained(
            self.base_model_path, 
            subfolder="scheduler"
        )

    def _init_generator(self):
        print(f"Loading Generator Enhance (Base: {self.base_model_path}, Rank: {self.lora_rank})...")
        self.G = DiffGANGeneratorEnhance(
            base_model_path=self.base_model_path, 
            lora_rank=self.lora_rank, 
            lora_modules=self.lora_modules, 
            weight_dtype=self.weight_dtype
        ).to(self.device, dtype=self.weight_dtype)

        if self.weight_path and os.path.exists(self.weight_path):
            print(f"Loading weights from {self.weight_path}...")
            # Load weights (Phase II saves full trainable state: conv_out + LoRA)
            raw_state = torch.load(self.weight_path, map_location="cpu")
            # Clean state dict keys if necessary (remove DDP wrappers)
            clean_state = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in raw_state.items()}
            
            # Load with strict=False because the model contains frozen weights not in the checkpoint
            # (Checkpoint only has conv_out and LoRA adapters)
            keys = self.G.load_state_dict(clean_state, strict=False)
            print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}, Unexpected keys: {len(keys.unexpected_keys)}")
            if len(keys.unexpected_keys) > 0:
                print(f"Sample unexpected: {keys.unexpected_keys[:5]}")
        else:
            print(f"Warning: Weight path '{self.weight_path}' not found. Using initialized weights.")

        self.G.eval().requires_grad_(False)
    
    # --- ADDED: Init Prompt Embedder ---
    def _init_prompt_embedder(self):
        """Initialize fixed prompt embedder to match training"""
        print(f"Loading Prompt Embeddings from {self.prompt_embeds_path}...")
        self.prompt_embedder = SDXLFixedPromptEmbedder(
            cache_path=self.prompt_embeds_path,
            device=self.device,
            dtype=self.weight_dtype
        )
        
    @torch.no_grad()
    def infer(
        self,
        lq: torch.Tensor,
        mask: torch.Tensor,
        return_type: Literal["pt", "np", "pil"] = "pt",
    ) -> Tuple[torch.Tensor | np.ndarray | List[Image.Image], List[Image.Image]]:
        """
        Perform object removal inference using Enhanced Generator
        """
        # Normalize input to [-1, 1] range
        lq = (lq * 2 - 1).float()
        
        # Encode images to latent space
        z_lq = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
        z_lq = z_lq * self.vae.config.scaling_factor
        bs = z_lq.shape[0]
        
        # Prepare mask latents (downsample user mask)
        scale_factor = 8 
        mask_latent = F.max_pool2d(mask.float(), kernel_size=scale_factor, stride=scale_factor).to(device=self.device, dtype=self.weight_dtype)
        
        # Generate consistent noise
        noise = torch.randn_like(z_lq, device=self.device, dtype=self.weight_dtype)
        
        # Generator timestep
        timesteps = torch.full((bs,), self.model_t, dtype=torch.long, device=self.device)

        z = self.scheduler.add_noise(z_lq, noise, timesteps)

        z_in = torch.cat([z, mask_latent, z_lq], dim=1)

        # --- MODIFIED: Get Embeddings via FixedPromptEmbedder ---
        prompt_embeds, pooled_prompt_embeds = self.prompt_embedder.get_embeddings(bs)
        
        # Time IDs
        height, width = lq.shape[-2:]
        time_ids = torch.tensor([
            [height, width, 0, 0, height, width]
        ], dtype=z_lq.dtype, device=z_lq.device).repeat(bs, 1)
        
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds, # Updated to use loaded embeddings
            "time_ids": time_ids
        }
        
        # Forward through generator
        # Phase II returns eps AND mask_logits
        # --- MODIFIED: Pass encoder_hidden_states (prompt_embeds) ---
        eps, mask_logits = self.G(
            z_in,
            timesteps,
            added_cond_kwargs=added_cond_kwargs,
            encoder_hidden_states=prompt_embeds
        )
        
        # Predicted Alpha
        alpha = torch.sigmoid(mask_logits)
        
        # Denoise
        z_out = self.scheduler.step(eps, timesteps, z).pred_original_sample

        z_out = z_out * alpha + z_lq * (1 - alpha)
        
        # VAE decode
        x = self.vae.decode(z_out.to(self.weight_dtype) / self.vae.config.scaling_factor).sample.float()
        x = (x + 1) / 2  # Convert back to [0, 1] range
        x = torch.clamp(x, 0, 1)

        # Process Alpha for return
        alpha_vis = F.interpolate(alpha.float(), size=(height, width), mode='bilinear')
        alpha_pils = [Image.fromarray((a[0].cpu().numpy() * 255).astype(np.uint8)) for a in alpha_vis]
        
        # Convert to requested output format
        if return_type == "pt":
            res = x
        elif return_type == "np":
            res = x.detach().cpu().numpy()
        elif return_type == "pil":
            res = [Image.fromarray(self._tensor2image(img)) for img in x]
            
        return res, alpha_pils

    @staticmethod
    def _tensor2image(tensor: torch.Tensor) -> np.ndarray:
        image = tensor.detach().cpu().clamp(0, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return np.transpose(image, (1, 2, 0))

def _calculate_resize_dimensions(width: int, height: int, resolution: int) -> Tuple[int, int]:
    target_short = resolution
    if min(width, height) < target_short:
        new_w = (width + 15) // 16 * 16
        new_h = (height + 15) // 16 * 16
    else:
        if width < height:
            new_w = target_short
            new_h = int(height * target_short / width)
        else:
            new_h = target_short
            new_w = int(width * target_short / height)
        new_w = (new_w + 15) // 16 * 16
        new_h = (new_h + 15) // 16 * 16
    return new_w, new_h

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DiffGAN Removal Inference (Enhance/Stage1)")
    parser.add_argument('-i', '--input_path', type=str, default='./inputs/imgs', 
                        help='Input image or folder')
    parser.add_argument('-m', '--mask_path', type=str, default='./inputs/masks',
                        help='Input mask image or folder')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
                        help='Output folder')
    parser.add_argument('-b', '--base_path', type=str, required=True, 
                        help='Path to SDXL base model')
    parser.add_argument('-w', '--weight_path', type=str, required=True, 
                        help='Path to trained model weights (.pth)')
    # --- ADDED: Prompt Embeds Path Argument ---
    parser.add_argument('-p', '--prompt_embeds_path', type=str, default="data/prompt_embeds", 
                        help='Path to directory containing cached prompt embeddings')
    parser.add_argument('-t', '--t', type=int, default=400, help='Model timestep')
    parser.add_argument('--resolution', type=int, default=512, help='Target resolution')
    parser.add_argument('--paste_back', action='store_true', 
                        help='Paste inpainted region back to original image')
    args = parser.parse_args()
    
    # ------------------------ input & output ------------------------
    if os.path.isfile(args.input_path):
        input_img_list = [args.input_path]
        result_root = args.output_path if args.output_path else 'results/test_single_img'
    else:
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = args.output_path if args.output_path else f'results/{os.path.basename(args.input_path.rstrip("/"))}'
        
    if os.path.isfile(args.mask_path):
        input_mask_list = [args.mask_path]
    else:
        input_mask_list = sorted(glob.glob(os.path.join(args.mask_path, '*.[jpJP][pnPN]*[gG]')))
        
    if len(input_img_list) == 0:
        print(f"No images found in {args.input_path}")
        sys.exit(0)

    if len(input_img_list) > 1 and len(input_img_list) != len(input_mask_list):
         print("Warning: Mismatch in number of images and masks. Attempting to match by filename...")
         mask_map = {os.path.splitext(os.path.basename(m))[0]: m for m in input_mask_list}
         matched_masks = []
         valid_imgs = []
         for img in input_img_list:
             base = os.path.splitext(os.path.basename(img))[0]
             if base in mask_map:
                 matched_masks.append(mask_map[base])
                 valid_imgs.append(img)
             else:
                 print(f"Skipping {base} (no matching mask found)")
         input_img_list = valid_imgs
         input_mask_list = matched_masks

    if len(input_img_list) == 0:
         print("No valid image-mask pairs found.")
         sys.exit(0)

    os.makedirs(result_root, exist_ok=True)
    test_img_num = len(input_img_list)
    print(f"Found {test_img_num} images to process.")
    
    # ------------------ set up pipeline -------------------
    # LoRA modules for Phase II (excluding conv_out as it is full fine-tuned)
    default_lora_modules = [
        "to_k", "to_q", "to_v", "to_out.0", "conv", "conv1",
        "conv2", "conv_shortcut", "proj_in", 
        "proj_out", "ff.net.2", "ff.net.0.proj"
    ]

    model = SDXLInferenceEnhance(
            base_model_path=args.base_path,
            weight_path=args.weight_path,
            prompt_embeds_path=args.prompt_embeds_path, # --- ADDED ---
            lora_modules=default_lora_modules,
            lora_rank=256,
            model_t=args.t,
            device="cuda" if torch.cuda.is_available() else "cpu",
            weight_dtype=torch.bfloat16
    )

    total_time = 0
    for i, (img_path, mask_path) in enumerate(zip(input_img_list, input_mask_list)):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            new_w, new_h = _calculate_resize_dimensions(image.width, image.height, args.resolution)
            image_r = image.resize((new_w, new_h), Image.BICUBIC)
            mask_r = mask.resize((new_w, new_h), Image.NEAREST)
        
            image_tensor = to_tensor(image_r).unsqueeze(0)
            mask_tensor = to_tensor(mask_r).unsqueeze(0)
            
            device = model.device
            image_tensor = image_tensor.to(device)
            mask_tensor = mask_tensor.to(device)
            
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            start_time = time.time()
            result_pil, alpha_pils = model.infer(lq=image_tensor, mask=mask_tensor, return_type="pil")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            
            result_pil = result_pil[0]
            alpha_pil = alpha_pils[0]
            
            result_pil = result_pil.resize(image.size, Image.BICUBIC)
            if args.paste_back:   
                alpha_pil = alpha_pil.resize(image.size, Image.BILINEAR)
                result_pil = Image.composite(result_pil, image, alpha_pil)
            
            save_path = os.path.join(result_root, f'{basename}.png')
            result_pil.save(save_path)
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f'\nAll results saved in {result_root}')
    if test_img_num > 0:
        print(f'Average Inference Time: {total_time / test_img_num:.4f}s')