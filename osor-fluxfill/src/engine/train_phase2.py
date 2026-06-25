import logging
import os
import shutil
from pathlib import Path
from contextlib import nullcontext
import importlib

import torch
import torch.nn.functional as F
from torch.serialization import get_unsafe_globals_in_checkpoint, add_safe_globals
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
import transformers
import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from PIL import Image
from safetensors.torch import load_file

# --- SRC Imports ---
from src.utils.common import instantiate_from_config, print_vram_state, SuppressLogging
from src.utils.tabulate import tabulate
from src.data.collate import ConsistentMaskCollate
from src.data.sampler import DistributedAspectRatioBucketSampler
from src.models.discriminator import SMMPatchGAN
from src.models.generator_enhance import FluxGeneratorEnhance
from src.models.components.prompt import FluxFixedPromptEmbedder
from src.utils.losses import MaskAwareGANLoss, MaskAwareL1Loss, LPIPSLoss, AlphaLoss

logger = get_logger(__name__, log_level="INFO")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

from src.engine.batch_input import BatchInput


class Phase2Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(config.training.seed)
        self.init_environment()
        self.init_models()
        self.summary_models()
        self.init_optimizers()
        self.init_dataset()
        self.prepare_all()
        
        logger.info(f"="*80)
        logger.info(f"Initializing Flux Removal Trainer - PHASE II (alpha-aware)")
        logger.info(f"="*80)

    def init_environment(self):
        logging_dir = Path(self.config.project.output_dir, self.config.project.logging_dir)
        accelerator_project_config = ProjectConfiguration(project_dir=self.config.project.output_dir, logging_dir=logging_dir)
        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.grad_accum_steps,
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
            mixed_precision=self.config.training.mixed_precision,
        )
        logger.info(accelerator.state, main_process_only=True)
        if accelerator.is_main_process:
            accelerator.init_trackers("train")
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_warning()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        if accelerator.is_main_process:
            if self.config.project.output_dir is not None:
                os.makedirs(self.config.project.output_dir, exist_ok=True)
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.accelerator = accelerator
        self.weight_dtype = weight_dtype
        self.device = accelerator.device

    def unwrap_model(self, model):
        model = self.accelerator.unwrap_model(model)
        return model

    def init_models(self):
        self.init_scheduler()
        self.init_vae()
        self.init_prompt_embedder()
        self.init_generator()
        self.init_discriminator()
        self.init_losses()

    def init_scheduler(self):
        """Initialize scheduler"""
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.config.model.flux_base, subfolder="scheduler"
        )

    def init_vae(self):
        self.vae = AutoencoderKL.from_pretrained(
            self.config.model.flux_base, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        self.vae.eval().requires_grad_(False)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def init_prompt_embedder(self):
        """Initialize fixed prompt embedder"""
        self.prompt_embedder = FluxFixedPromptEmbedder(
            cache_path=self.config.model.prompt_embeds_path,
            device=self.device,
            dtype=self.weight_dtype
        )

    def init_losses(self):
        """Initialize loss functions"""
        # LPIPS
        self.net_lpips = LPIPSLoss(device=self.device)
        # GAN Loss
        self.gan_loss = MaskAwareGANLoss()
        # L1 Loss
        self.l1_loss = MaskAwareL1Loss()
        # Alpha Loss
        self.alpha_loss = AlphaLoss(
            lambda_bce=self.config.loss_weights.alpha_bce,
            lambda_dice=self.config.loss_weights.alpha_dice,
        ).to(self.device)

    def init_generator(self):
        """Initialize Flux DiT generator with LoRA configuration"""
        logger.info(f"Loading base model from: {self.config.model.flux_base}")
        self.G = FluxGeneratorEnhance(
            base_model_path=self.config.model.flux_base,
            lora_rank=self.config.model.lora.rank,
            lora_modules=self.config.model.lora.modules,
            weight_dtype=self.weight_dtype
        ).to(self.device)
        if self.config.training.grad_checkpointing:
            logger.info("Enabling gradient checkpointing for memory efficiency")
            self.G.enable_gradient_checkpointing()

        lora_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        for p in lora_params:
            p.data = p.to(torch.float32)
        logger.info(f"Forced {len(lora_params)} LoRA parameters to float32.")

    def init_discriminator(self):
        ctx = (
            nullcontext()
            if self.accelerator.is_local_main_process
            else SuppressLogging(logging.WARNING)
        )
        with ctx:
            self.D = SMMPatchGAN(backbone_path=self.config.model.clip_backbone).to(device=self.device)
        self.D.train().requires_grad_(True)

    def summary_models(self):
        table_data = []
        for attr, value in self.__dict__.items():
            if not isinstance(value, torch.nn.Module):
                continue
            model = value
            model_type = type(model).__name__
            total_params = sum(p.numel() for p in model.parameters()) / 1_000_000
            learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
            table_data.append([attr, model_type, f"{total_params:.2f}", f"{learnable_params:.2f}"])
        headers = ["Model Name", "Model Type", "Total Parameters (M)", "Learnable Parameters (M)"]
        table = tabulate(table_data, headers=headers, tablefmt="pretty")
        logger.info(f"Model Summary:\n{table}")

    def init_optimizers(self):
        logger.info(f"Creating {self.config.training.optimizer.type} optimizers")
        if self.config.training.optimizer.type == "adam":
            optimizer_cls = torch.optim.AdamW
        elif self.config.training.optimizer.type == "rmsprop":
            optimizer_cls = torch.optim.RMSprop
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.training.optimizer.type}")
        self.G_params = list(filter(lambda p: p.requires_grad, self.G.parameters()))
        self.G_opt = optimizer_cls(
            self.G_params,
            lr=self.config.training.optimizer.lr_g,
            betas=self.config.training.optimizer.betas,
        )
        self.D_params = list(filter(lambda p: p.requires_grad, self.D.parameters()))
        self.D_opt = optimizer_cls(
            self.D_params,
            lr=self.config.training.optimizer.lr_d,
            betas=self.config.training.optimizer.betas,
        )

    def init_dataset(self):
        dataset = instantiate_from_config(self.config.data)
        
        aug_flag = getattr(dataset, 'augment_mask', False)
        
        batch_sampler = DistributedAspectRatioBucketSampler(
            dataset,
            batch_size=self.config.data.batch_size,
            drop_last=True
        )

        collate_fn = ConsistentMaskCollate() if aug_flag else None
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
        )
        
        self.batch_transform = instantiate_from_config(self.config.batch)

    def prepare_all(self):
        logger.info("Wrapping models, optimizers and dataloaders")
        attrs = ["G", "D", "G_opt", "D_opt", "dataloader"]
        prepared_objs = self.accelerator.prepare(*[getattr(self, attr) for attr in attrs])
        for attr, obj in zip(attrs, prepared_objs):
            setattr(self, attr, obj)
        print_vram_state("After accelerator.prepare", logger=logger)

    def force_optimizer_ckpt_safe(self, checkpoint_dir):
        def get_symbol(s):
            module_name, symbol_name = s.rsplit('.', 1)
            module = importlib.import_module(module_name)
            symbol = getattr(module, symbol_name)
            return symbol

        for file_name in os.listdir(checkpoint_dir):
            if "optimizer" in file_name and not file_name.endswith("safetensors"):
                path = os.path.join(checkpoint_dir, file_name)
                unsafe_globals = get_unsafe_globals_in_checkpoint(path)
                logger.info(f"Unsafe globals in {path}: {unsafe_globals}")
                unsafe_globals = list(map(get_symbol, unsafe_globals))
                add_safe_globals(unsafe_globals)

    def attach_accelerator_hooks(self):
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(self.unwrap_model(model), FluxGeneratorEnhance):
                        weights.pop(i)
                        unwrapped = self.unwrap_model(model)
                        state_dict = {n: p.detach().clone().data for n, p in unwrapped.named_parameters() if p.requires_grad}
                        torch.save(state_dict, os.path.join(output_dir, "state_dict.pth"))
                        break

        def load_model_hook(models, input_dir):
            for i, model in enumerate(models):
                 if isinstance(self.unwrap_model(model), FluxGeneratorEnhance):
                    model = models.pop(i)
                    state_dict_path = os.path.join(input_dir, "state_dict.pth")
                    if os.path.exists(state_dict_path):
                        state_dict = torch.load(state_dict_path)
                        m, u = model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Loading lora parameters, unexpected keys: {len(u) if u else 0}")
                    else:
                        logger.warning(f"state_dict.pth not found at {state_dict_path}")
                    break

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def on_training_start(self):
        global_step = 0
        if self.config.checkpointing.resume_from_checkpoint:
            path = self.config.checkpointing.resume_from_checkpoint
            self.force_optimizer_ckpt_safe(path)
            self.accelerator.load_state(path)
            global_step = int(path.split("-")[-1])
            
        elif self.config.checkpointing.init_checkpoint:
            ckpt_path = self.config.checkpointing.init_checkpoint
            g_path = os.path.join(ckpt_path, "state_dict.pth")
            state_dict_g = torch.load(g_path, map_location="cpu")

            keys_to_pop = []
            for k in list(state_dict_g.keys()):
                if "proj_out" in k:
                    target = self.unwrap_model(self.G).transformer
                    if hasattr(target, "base_model"): target = target.base_model.model
                    
                    if "weight" in k:
                        with torch.no_grad():
                             target.proj_out.weight.data[:64, :] = state_dict_g[k]
                        keys_to_pop.append(k)
                    elif "bias" in k:
                        with torch.no_grad():
                             target.proj_out.bias.data[:64] = state_dict_g[k]
                        keys_to_pop.append(k)
            
            for k in keys_to_pop:
                del state_dict_g[k]

            self.unwrap_model(self.G).load_state_dict(state_dict_g, strict=False)
            
            d_path = os.path.join(ckpt_path, "model.safetensors") 
            if os.path.exists(d_path):
                self.unwrap_model(self.D).load_state_dict(load_file(d_path), strict=True)

        self.global_step = global_step
        self.pbar = tqdm(
            range(0, self.config.training.max_steps),
            initial=global_step,
            desc="Steps",
            disable=not self.accelerator.is_main_process,
        )

    def prepare_batch_inputs(self, batch):
        batch = self.batch_transform(batch)
        shot = (batch["shot"] * 2 - 1).float()
        bg = (batch["bg"] * 2 - 1).float()
        mask = batch["mask"] 
        gt_mask = batch["gt_mask"]
        
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
            packed_latents_shot = FluxGeneratorEnhance.pack_latents(latents_shot, bs, self.vae.config.latent_channels, H_lat, W_lat)

        mask_tensor = mask.to(self.device, dtype=self.weight_dtype)
        mask_tensor = mask_tensor[:, 0, :, :]
        mask_tensor = mask_tensor.view(bs, H_lat, self.vae_scale_factor, W_lat, self.vae_scale_factor)
        mask_tensor = mask_tensor.permute(0, 2, 4, 1, 3)
        mask_tensor = mask_tensor
        mask_tensor = mask_tensor.reshape(bs, self.vae_scale_factor * self.vae_scale_factor, H_lat, W_lat)
        packed_mask = FluxGeneratorEnhance.pack_latents(mask_tensor, bs, self.vae_scale_factor * self.vae_scale_factor, H_lat, W_lat)
        
        noise = torch.randn_like(packed_latents_shot)
        t_val = self.config.model.model_t / 1000.0  
        timestep = torch.full((bs,), t_val, device=self.device, dtype=self.weight_dtype)
        z_t_packed = (1 - t_val) * packed_latents_shot + t_val * noise

        prompt_embeds, pooled_prompt_embeds, txt_ids = self.prompt_embedder.get_embeddings(bs)

        mask_latent = F.max_pool2d(mask, kernel_size=self.vae_scale_factor, stride=self.vae_scale_factor).to(device=self.device, dtype=self.weight_dtype)
        gt_mask_latent = F.max_pool2d(gt_mask, kernel_size=self.vae_scale_factor, stride=self.vae_scale_factor).to(device=self.device, dtype=self.weight_dtype)

        self.batch_inputs = BatchInput(
            shot=shot, bg=bg, mask=mask, gt_mask=gt_mask,
            latents_shot=latents_shot,
            mask_latent=mask_latent, 
            gt_mask_latent=gt_mask_latent,
            packed_latents_shot=packed_latents_shot, 
            packed_mask=packed_mask, 
            z_t_packed=z_t_packed, 
            timestep=timestep,
            img_ids=img_ids,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            txt_ids=txt_ids,
            H_lat=H_lat, W_lat=W_lat
        )

    def forward_generator(self):
        hidden_states = torch.cat([self.batch_inputs.z_t_packed, self.batch_inputs.packed_latents_shot, self.batch_inputs.packed_mask], dim=2)

        guidance_scale = self.config.model.guidance_scale
        bs = self.batch_inputs.z_t_packed.shape[0]
        guidance = torch.full((bs,), guidance_scale, device=self.device, dtype=torch.float32)

        v_pred, mask_logits = self.G(
            hidden_states=hidden_states,
            timestep=self.batch_inputs.timestep,
            encoder_hidden_states=self.batch_inputs.prompt_embeds,
            pooled_projections=self.batch_inputs.pooled_prompt_embeds,
            txt_ids=self.batch_inputs.txt_ids,
            img_ids=self.batch_inputs.img_ids,
            guidance=guidance,  
            return_dict=False,
        )

        mask_logits= FluxGeneratorEnhance.unpack_latents(
            mask_logits,
            self.batch_inputs.H_lat,
            self.batch_inputs.W_lat
        )
        alpha = torch.sigmoid(mask_logits)

        t = self.batch_inputs.timestep.view(-1, 1, 1)
        pred_packed_latents = self.batch_inputs.z_t_packed - t * v_pred

        pred_latents = FluxGeneratorEnhance.unpack_latents(
            pred_packed_latents, 
            self.batch_inputs.H_lat, 
            self.batch_inputs.W_lat
        )

        pred_latents = (pred_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents_shot = (self.batch_inputs.latents_shot / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        
        composite_latents = pred_latents * alpha + latents_shot * (1 - alpha)

        x = self.vae.decode(composite_latents.to(self.weight_dtype)).sample.float()
        
        return x, mask_logits, alpha

    def optimize_generator(self):
        with self.accelerator.accumulate(self.G):
            self.unwrap_model(self.D).eval().requires_grad_(False)
            x, mask_logits, alpha = self.forward_generator() 
            
            if not torch.isfinite(x).all(): 
                logger.warning(f"Non-finite values detected in x at step {self.global_step}")
                raise RuntimeError("Non-finite encountered")

            alpha_vis = F.interpolate(alpha, size=self.batch_inputs.mask.shape[-2:], mode='area')
            
            self.G_pred = x
            self.alpha_vis = alpha_vis
            
            # Updated keys: x vs bg
            loss_l1 = self.l1_loss(x, self.batch_inputs.bg, self.batch_inputs.gt_mask) * self.config.loss_weights.l1
            loss_lpips = self.net_lpips(x, self.batch_inputs.bg) * self.config.loss_weights.lpips

            fake_logits, _ = self.D(x)
            loss_disc = self.gan_loss.generator_loss(fake_logits, self.batch_inputs.gt_mask) * self.config.loss_weights.gan
            
            loss_alpha, loss_alpha_dict = self.alpha_loss(mask_logits, alpha, self.batch_inputs.gt_mask_latent)

            loss_G = loss_l1 + loss_lpips + loss_disc + loss_alpha
            self.accelerator.backward(loss_G)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_((p for p in self.unwrap_model(self.G).parameters() if p.requires_grad), self.config.training.max_grad_norm)
            self.G_opt.step()
            self.G_opt.zero_grad()
        
        return dict(G_total=loss_G, G_l1=loss_l1, G_lpips=loss_lpips, G_disc=loss_disc, G_alpha=loss_alpha, G_alpha_bce=loss_alpha_dict["bce"], G_alpha_dice=loss_alpha_dict["dice"])

    def optimize_discriminator(self):
        # Updated keys: bg
        bg = self.batch_inputs.bg
        with torch.no_grad():
            x, _, alpha = self.forward_generator()

        if not torch.isfinite(x).all(): 
                logger.warning(f"Non-finite values detected in x at step {self.global_step}")
                raise RuntimeError("Non-finite encountered")

        alpha_vis = F.interpolate(alpha, size=self.batch_inputs.mask.shape[-2:], mode='area')

        self.G_pred = x
        self.alpha_vis = alpha_vis

        with self.accelerator.accumulate(self.D):
            self.unwrap_model(self.D).train().requires_grad_(True)
            real_logits, real_features = self.D(bg)
            fake_logits, _ = self.D(x)
            loss_gan = self.gan_loss.discriminator_loss(real_logits, fake_logits, self.batch_inputs.gt_mask)

            loss_gp = self.gan_loss.gradient_penalty(
                real_features, 
                self.unwrap_model(self.D).heads
            ) * self.config.loss_weights.gp

            loss_D = loss_gan + loss_gp
            
            if not torch.isfinite(loss_D).all():
                logger.error("Non-finite loss at step %d", self.global_step)
                raise RuntimeError("Non-finite encountered")
                
            self.accelerator.backward(loss_D)
            self.D_opt.step()
            self.D_opt.zero_grad()
        
        loss_dict = dict(D=loss_D, D_gan=loss_gan, D_gp=loss_gp)
        with torch.no_grad():
            real_logits = torch.tensor([logit_map.mean() for logit_map in real_logits], device=self.device).mean()
            fake_logits = torch.tensor([logit_map.mean() for logit_map in fake_logits], device=self.device).mean()
        loss_dict.update(dict(D_logits_real=real_logits, D_logits_fake=fake_logits))
        return loss_dict

    def run(self):
        self.attach_accelerator_hooks()
        self.on_training_start()
        self.batch_count = 0
        
        logger.info("Starting Phase II Training (alpha-aware removal)")
        
        current_epoch = 0
        while self.global_step < self.config.training.max_steps:
            self.dataloader.batch_sampler.batch_sampler.set_epoch(current_epoch)
            current_epoch += 1
            train_loss = {}
            for batch in self.dataloader:
                self.prepare_batch_inputs(batch)
                bs = len(self.batch_inputs.shot) # Updated
                generator_step = ((self.batch_count // self.config.training.grad_accum_steps) % 2) == 0
                if generator_step:
                    loss_dict = self.optimize_generator()
                else:
                    loss_dict = self.optimize_discriminator()

                for k, v in loss_dict.items():
                    avg_loss = self.accelerator.gather(v.repeat(bs)).mean()
                    if k not in train_loss:
                        train_loss[k] = 0
                    train_loss[k] += avg_loss.item() / self.config.training.grad_accum_steps

                self.batch_count += 1
                if self.accelerator.sync_gradients:
                    state = "Generator Step" if generator_step else "Discriminator Step"
                    _, _, peak = print_vram_state(None)
                    self.pbar.set_description(f"{state}, VRAM peak: {peak:.2f} GB")

                if self.accelerator.sync_gradients and not generator_step:
                    self.global_step += 1
                    self.pbar.update(1)
                    self.accelerator.log(train_loss, step=self.global_step)
                    train_loss = {}
                    if self.global_step % self.config.logging.log_image_steps == 0 or self.global_step == 1:
                        self.log_images()
                    if self.global_step % self.config.logging.log_grad_steps == 0 or self.global_step == 1:
                        self.log_grads()
                    if self.global_step % self.config.checkpointing.checkpointing_steps == 0 or self.global_step == 1:
                        self.save_checkpoint()

                if self.global_step >= self.config.training.max_steps:
                    break
        self.accelerator.end_training()

    def log_images(self):
        N = 4
        image_logs = dict(
            shot=(self.batch_inputs.shot[:N] + 1) / 2,
            bg=(self.batch_inputs.bg[:N] + 1) / 2,
            G=(self.G_pred[:N] + 1) / 2,
            mask=(self.batch_inputs.mask[:N].repeat(1, 3, 1, 1)),
            gt_mask=(self.batch_inputs.gt_mask[:N].repeat(1, 3, 1, 1)),
            alpha=(self.alpha_vis[:N].repeat(1, 3, 1, 1)),
        )

        if not self.accelerator.is_main_process:
            return

        for tracker in self.accelerator.trackers:
            if tracker.name == "tensorboard":
                for tag, images in image_logs.items():
                    tracker.writer.add_image(
                        f"image/{tag}",
                        make_grid(images.float(), nrow=4),
                        self.global_step,
                    )

        for key, images in image_logs.items():
            image_arrs = (images * 255.0).clamp(0, 255).to(torch.uint8) \
                .permute(0, 2, 3, 1).contiguous().cpu().numpy()
            save_dir = os.path.join(
                self.config.project.output_dir, self.config.project.logging_dir, "log_images", f"{self.global_step:07}", key)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for i, img in enumerate(image_arrs):
                Image.fromarray(img).save(os.path.join(save_dir, f"sample{i}.png"))

    def log_grads(self):
        self.unwrap_model(self.D).eval().requires_grad_(False)
        x, mask_logits, alpha = self.forward_generator()

        # Use gt_mask (m_gt) to match the active training path; the diagnostic
        # previously used alpha_vis, which logged gradients for a different loss.
        loss_l1 = self.l1_loss(x, self.batch_inputs.bg, self.batch_inputs.gt_mask) * self.config.loss_weights.l1
        loss_lpips = self.net_lpips(x, self.batch_inputs.bg) * self.config.loss_weights.lpips
        fake_logits, _ = self.D(x)
        loss_disc = self.gan_loss.generator_loss(fake_logits, self.batch_inputs.gt_mask) * self.config.loss_weights.gan
        loss_alpha, _ = self.alpha_loss(mask_logits, alpha, self.batch_inputs.gt_mask_latent)

        losses = [("l1", loss_l1), ("lpips", loss_lpips), ("disc", loss_disc), ("alpha", loss_alpha)]
        grad_dict = {}
        self.G_opt.zero_grad()
        for idx, (name, loss) in enumerate(losses):
            retain_graph = idx != len(losses) - 1
            loss.backward(retain_graph=retain_graph)
            lora_module_grads = {}
            for module_name, module in self.unwrap_model(self.G).named_modules():
                for suffix in self.config.logging.log_grad_modules:
                    if module_name.endswith(suffix):
                        grads_list = [
                            p.grad.flatten() for p in module.parameters() 
                            if p.requires_grad and p.grad is not None
                        ]
                        if len(grads_list) > 0:
                            flat_grad = torch.cat(grads_list)
                            lora_module_grads.setdefault(suffix, []).append(flat_grad)
                        break
            for k, v in lora_module_grads.items():
                grad_dict[f"grad_norm/{k}_{name}"] = torch.norm(torch.cat(v)).item()
            self.G_opt.zero_grad()
        self.accelerator.log(grad_dict, step=self.global_step)

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            if self.config.checkpointing.checkpoints_total_limit is not None:
                checkpoints = os.listdir(self.config.project.output_dir)
                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                if len(checkpoints) >= self.config.checkpointing.checkpoints_total_limit:
                    num_to_remove = len(checkpoints) - self.config.checkpointing.checkpoints_total_limit + 1
                    removing_checkpoints = checkpoints[0:num_to_remove]
                    logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                    for removing_checkpoint in removing_checkpoints:
                        removing_checkpoint = os.path.join(self.config.project.output_dir, removing_checkpoint)
                        shutil.rmtree(removing_checkpoint)
            save_path = os.path.join(self.config.project.output_dir, f"checkpoint-{self.global_step}")
            self.accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")