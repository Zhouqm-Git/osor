import torch
import torch.nn as nn
import open_clip
import os

class CLIPConvNextBackbone(nn.Module):
    """
    Wrapper for OpenCLIP ConvNeXt backbone.
    Fixes the hardcoded path issue by accepting a path argument or downloading.
    """
    def __init__(self, model_path=None, precision="bf16"):
        super().__init__()
        
        # Determine loading strategy
        pretrained_arg = model_path if (model_path and os.path.exists(model_path)) else "laion2b_s34b_b82k_augreg_soup"
        
        print(f"Initializing CLIP Backbone. Loading from: {pretrained_arg}")
        
        try:
            self.model, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge",
                pretrained=pretrained_arg,
                precision=precision,
            )
        except Exception as e:
            print(f"Error loading specific weight. Fallback to default download. Error: {e}")
            self.model, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge",
                pretrained="laion2b_s34b_b82k_augreg_soup",
                precision=precision,
            )
            
        self.model.eval().requires_grad_(False)

    def _get_visual_stages(self, image):
        """Extract intermediate features"""
        intermediates = self.model.visual.trunk.forward_intermediates(
            image,
            indices=None,
            norm=False,
            stop_early=False,
            intermediates_only=True,
        )
        return intermediates[1:]

    def encode_image(self, image):
        return self._get_visual_stages(image)

    def forward(self, x):
        return self.encode_image(x)
