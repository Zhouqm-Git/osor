import torch
import torch.nn as nn
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from src.models.backbone import CLIPConvNextBackbone


class MultiLevelDConv(nn.Module):
    """Lightweight Multi-scale heads"""
    def __init__(self, in_channels_list, out_ch=512):
        super().__init__()
        self.heads = nn.ModuleList()
        for in_ch in in_channels_list:
            self.heads.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1)),
                nn.LeakyReLU(0.2, True),
                BlurPool(out_ch, filt_size=5, pad_type="reflect", stride=2),
                spectral_norm(nn.Conv2d(out_ch, 1, 1))
            ))
            
    def forward(self, features):
        logits = []
        for head, feat in zip(self.heads, features):
            logits.append(head(feat).squeeze(1))
        return logits

class SMMPatchGAN(nn.Module):
    """
    Soft-Mask Multi-scale PatchGAN.
    Uses Frozen CLIP-ConvNeXt backbone + Trainable Heads.
    """
    def __init__(self, backbone_path=None):
        super().__init__()
        self.backbone = CLIPConvNextBackbone(model_path=backbone_path)
        # ConvNeXt feature channels: [384, 768, 1536, 3072]
        self.heads = MultiLevelDConv([384, 768, 1536, 3072])
        
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))

    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W] in range [-1, 1]
        Returns:
            logits: List of tensors from each head
            features: Intermediate features (for feature matching loss)
        """
        # Normalize to CLIP space
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std
        
        features = self.backbone(x)
        logits = self.heads(features)
        
        return logits, features

    def train(self, mode=True):
        """Overridden to keep backbone in eval mode"""
        self.backbone.eval()
        self.heads.train(mode)
        return self

    def eval(self):
        """Overridden to keep backbone in eval mode"""
        self.backbone.eval()
        self.heads.eval()
        return self

    def requires_grad_(self, requires_grad=True):
        """Overridden to keep backbone frozen"""
        self.backbone.requires_grad_(False)
        self.heads.requires_grad_(requires_grad)
        return self
