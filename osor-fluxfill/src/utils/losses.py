import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class MaskAwareGANLoss(nn.Module):
    """
    Soft-Mask GAN Loss.
    Uses area-downsampled mask as soft labels for the discriminator.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def discriminator_loss(self, real_logits, fake_logits, mask):
        """
        Args:
            real_logits: List of tensors [B, 1, h, w]
            fake_logits: List of tensors [B, 1, h, w]
            mask: [B, 1, H, W] - high res mask
        """
        B = mask.shape[0]
        loss = 0.0
        for real, fake in zip(real_logits, fake_logits):
            # Downsample mask to feature size using Area (Soft Label)
            h, w = fake.shape[-2:]
            soft_mask = F.interpolate(mask, size=(h, w), mode='area').squeeze(1)
            target = 1.0 - soft_mask
            
            # Uniform per-pixel mean over each logit map.
            real_loss = self.bce(real, torch.ones_like(real)).view(B, -1).mean(1).mean()
            fake_loss = self.bce(fake, target).view(B, -1).mean(1).mean()
            
            loss += (real_loss + fake_loss)
        
        return loss

    def generator_loss(self, fake_logits, mask):
        B = mask.shape[0]
        loss = 0.0
        for fake in fake_logits:
            h, w = fake.shape[-2:]
            soft_mask = F.interpolate(mask, size=(h, w), mode='area').squeeze(1)
            
            pixel_loss = self.bce(fake, torch.ones_like(fake))
            
            # Mask-area-normalized (per-sample) non-saturating loss.
            weighted = (pixel_loss * soft_mask).view(B, -1).sum(1)
            occ_sum = soft_mask.view(B, -1).sum(1)
            occ_sum_clamped = occ_sum.clamp_min(1e-4)
            
            loss += (weighted / occ_sum_clamped).mean()
            
        return loss

    def gradient_penalty(self, features, heads):
        total_gp = 0.0
        features = [f.detach().requires_grad_(True) for f in features]
        head_layers = heads.heads
        for feat, head in zip(features, head_layers):
            logits = head(feat)

            grad = torch.autograd.grad(
                outputs=logits.sum(),
                inputs=feat,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            # R1 on head inputs: per-element squared-gradient mean, averaged across scales.
            gp = (grad ** 2).mean()
            total_gp += gp

        return total_gp / len(features)

class MaskAwareL1Loss(nn.Module):
    """
    Mask-aware L1 Loss.
    Computes L1 loss only in the masked region and normalizes by mask area.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: [B, C, H, W] - predicted image
            target: [B, C, H, W] - ground truth image
            mask: [B, 1, H, W] - mask (1 for hole, 0 for background)
        """
        # Per-sample normalization then uniform batch mean, matching
        # L_rec = E[ ||w*(x_hat - x_bg)||_1 / (||w||_1 + eps) ].
        diff = torch.abs(mask * (pred - target))
        num = diff.flatten(1).sum(1)                       # [B]
        den = mask.flatten(1).sum(1).clamp_min(1e-4)       # [B]
        loss = (num / den).mean()
        return loss

class AlphaLoss(nn.Module):
    def __init__(self, lambda_bce=1.0, lambda_dice=2.0):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred_logits, pred_alpha, gt_alpha):
        loss_bce = self.bce(pred_logits, gt_alpha)
        
        # Dice
        smooth = 1.0
        iflat = pred_alpha.view(-1)
        tflat = gt_alpha.view(-1)
        intersection = (iflat * tflat).sum()
        loss_dice = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        
        total_loss = self.lambda_bce * loss_bce + self.lambda_dice * loss_dice
        
        return total_loss, {"bce": self.lambda_bce * loss_bce, "dice": self.lambda_dice * loss_dice}

class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lpips = lpips.LPIPS(net='vgg').to(device).eval()
        for param in self.lpips.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        return self.lpips(x, y).mean()