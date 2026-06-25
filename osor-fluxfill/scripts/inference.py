#!/usr/bin/env python3
"""
DiffGAN-Removal Inference Script (Flux Version)
Phase I: Standard Removal (Hard Blending)
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
from typing import Literal, List, Tuple
from torchvision.transforms.functional import to_tensor
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from scipy.ndimage import distance_transform_edt

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.generator import FluxGenerator
from src.models.components.prompt import FluxFixedPromptEmbedder

class FluxInference:
    """
    Flux inference class for object removal tasks (Phase I).
    """
    
    def __init__(
        self,
        base_model_path: str,
        weight_path: str,
        prompt_embeds_path: str,
        lora_modules: List[str],
        lora_rank: int,
        model_t: int,
        guidance_scale: float,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16
    ):
        self.base_model_path = base_model_path
        self.weight_path = weight_path
        self.prompt_embeds_path = prompt_embeds_path
        self.lora_modules = lora_modules
        self.lora_rank = lora_rank
        self.model_t = model_t
        self.guidance_scale = guidance_scale
        self.device = device
        self.weight_dtype = weight_dtype

        # Initialize models
        self._init_vae()
        self._init_scheduler()
        self._init_generator()
        self._init_prompt_embedder()
                
    def _init_vae(self):
        print(f"Loading VAE from {self.base_model_path}...")
        self.vae = AutoencoderKL.from_pretrained(
            self.base_model_path, 
            subfolder="vae", 
            torch_dtype=self.weight_dtype
        ).to(self.device)
        self.vae.eval().requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
    def _init_scheduler(self):
        print(f"Loading Scheduler from {self.base_model_path}...")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.base_model_path, 
            subfolder="scheduler"
        )

    def _init_generator(self):
        print(f"Loading Flux Generator (Base: {self.base_model_path}, Rank: {self.lora_rank})...")
        self.G = FluxGenerator(
            base_model_path=self.base_model_path, 
            lora_rank=self.lora_rank, 
            lora_modules=self.lora_modules, 
            weight_dtype=self.weight_dtype
        ).to(self.device)

        if self.weight_path and os.path.exists(self.weight_path):
            print(f"Loading weights from {self.weight_path}...")
            # Load state dict (LoRA weights)
            state_dict = torch.load(self.weight_path, map_location="cpu")
            self.G.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: Weight path '{self.weight_path}' not found. Using initialized weights.")

        self.G.eval().requires_grad_(False)

    def _init_prompt_embedder(self):
        print(f"Loading Prompt Embeddings from {self.prompt_embeds_path}...")
        self.prompt_embedder = FluxFixedPromptEmbedder(
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
    ) -> torch.Tensor | np.ndarray | List[Image.Image]:
        """
        Perform object removal inference using Flux
        """
        shot = (lq * 2 - 1).float()
        bs = shot.shape[0]
        height, width = shot.shape[-2:]
        H_lat = height // self.vae_scale_factor
        W_lat = width // self.vae_scale_factor

        packed_h = H_lat // 2
        packed_w = W_lat // 2
        
        img_ids = torch.zeros(packed_h, packed_w, 3, device=self.device, dtype=self.weight_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_h, device=self.device)[:, None]  # Y
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_w, device=self.device)[None, :]  # X
        img_ids = img_ids.reshape(-1, 3)
        
        with torch.no_grad():
            latents_shot = self.vae.encode(shot.to(self.device, dtype=self.weight_dtype)).latent_dist.sample()
            latents_shot = (latents_shot - self.vae.config.shift_factor) * self.vae.config.scaling_factor
            packed_latents_shot = FluxGenerator.pack_latents(latents_shot, bs, self.vae.config.latent_channels, H_lat, W_lat)

        mask_tensor = mask.to(self.device, dtype=self.weight_dtype) 
        mask_tensor = mask_tensor[:, 0, :, :]
        mask_tensor = mask_tensor.view(bs, H_lat, self.vae_scale_factor, W_lat, self.vae_scale_factor)
        mask_tensor = mask_tensor.permute(0, 2, 4, 1, 3)
        mask_tensor = mask_tensor.reshape(bs, self.vae_scale_factor * self.vae_scale_factor, H_lat, W_lat)
        packed_mask = FluxGenerator.pack_latents(mask_tensor, bs, self.vae_scale_factor * self.vae_scale_factor, H_lat, W_lat)
        
        noise = torch.randn_like(packed_latents_shot)
        t_val = self.model_t / 1000.0  
        timestep = torch.full((bs,), t_val, device=self.device, dtype=self.weight_dtype)
        z_t_packed = (1 - t_val) * packed_latents_shot + t_val * noise

        prompt_embeds, pooled_prompt_embeds, txt_ids = self.prompt_embedder.get_embeddings(bs)

        mask_latent = F.max_pool2d(mask, kernel_size=self.vae_scale_factor, stride=self.vae_scale_factor).to(device=self.device, dtype=self.weight_dtype)

        hidden_states = torch.cat([z_t_packed, packed_latents_shot, packed_mask], dim=2)
        
        guidance = torch.full((bs,), self.guidance_scale, device=self.device, dtype=torch.float32)

        v_pred = self.G(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=guidance,  
            return_dict=False,
        )
        
        t = timestep.view(-1, 1, 1)
        pred_packed_latents = z_t_packed - t * v_pred

        pred_latents = FluxGenerator.unpack_latents(
            pred_packed_latents, 
            H_lat, 
            W_lat
        )

        pred_latents = (pred_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        latents_shot = (latents_shot / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        pred_latents = pred_latents * mask_latent + latents_shot * (1 - mask_latent)

        x = self.vae.decode(pred_latents.to(self.weight_dtype)).sample.float()
        
        # Post-process
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = torch.clamp(x, 0, 1)
        
        if return_type == "pt":
            return x
        elif return_type == "np":
            return x.detach().cpu().numpy()
        elif return_type == "pil":
            return [Image.fromarray(self._tensor2image(img)) for img in x]
    
    @staticmethod
    def _tensor2image(tensor: torch.Tensor) -> np.ndarray:
        image = tensor.detach().cpu().clamp(0, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return np.transpose(image, (1, 2, 0))

def _calculate_resize_dimensions(width: int, height: int) -> Tuple[int, int]:
    # FLUX specific: 16x alignment
    new_w = (width + 15) // 16 * 16
    new_h = (height + 15) // 16 * 16
    return new_w, new_h


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DiffGAN Removal Inference (Flux - Enhance)")
    parser.add_argument('-i', '--input_path', type=str, default='./inputs/imgs', help='Input image or folder')
    parser.add_argument('-m', '--mask_path', type=str, default='./inputs/masks', help='Input mask or folder')
    parser.add_argument('-o', '--output_path', type=str, default=None, help='Output folder')
    parser.add_argument('-b', '--base_path', type=str, required=True, help='Path to Flux base model')
    parser.add_argument('-w', '--weight_path', type=str, required=True, help='Path to weights (.pth)')
    parser.add_argument('-p', '--prompt_embeds_path', type=str, required=True, help='Path to fixed prompt embeddings (.pt)')
    parser.add_argument('-t', '--t', type=int, default=400, help='Model timestep (e.g. 400)')
    parser.add_argument('-g', '--g', type=float, default=30.0, help='Guidance scale (e.g. 30.0)')    
    args = parser.parse_args()
    
    lora_modules = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2", "proj_mlp", "single_transformer_blocks.*.proj_out"]
    
    # ------------------------ input & output ------------------------
    # Handle single file or directory inputs
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

    # Ensure equal number of masks if processing multiple
    if len(input_img_list) > 1 and len(input_img_list) != len(input_mask_list):
         # Try matching by filename if lists differ
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
    
    model = FluxInference(
        base_model_path=args.base_path,
        weight_path=args.weight_path,
        prompt_embeds_path=args.prompt_embeds_path,
        lora_modules=lora_modules,
        lora_rank=64, 
        model_t=args.t,
        guidance_scale=args.g,
        device="cuda"
    )

    total_time = 0
    for i, (img_path, mask_path) in enumerate(zip(input_img_list, input_mask_list)):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
            
        new_w, new_h = _calculate_resize_dimensions(image.width, image.height)
        image_r = image.resize((new_w, new_h), Image.BICUBIC)
        mask_r = mask.resize((new_w, new_h), Image.NEAREST)
        
        device = model.device
        image_t = to_tensor(image_r).unsqueeze(0).to(device)
        mask_t = to_tensor(mask_r).unsqueeze(0).to(device)

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start_time = time.time()
        res_pil = model.infer(image_t, mask_t, return_type="pil")[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        total_time += elapsed

        res_pil = res_pil.resize(image.size, Image.BICUBIC)            
        res_pil.save(os.path.join(result_root, f"{basename}.png"))

    print(f'\nAll results saved in {result_root}')
    if test_img_num > 0:
        print(f'Average Inference Time: {total_time / test_img_num:.4f}s')
