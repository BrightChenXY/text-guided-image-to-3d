import torch
import torch.nn as nn


class TextImageConditionProjector(nn.Module):
    """
    Trainable projection for CLIP text tokens (768 -> 1024 by default).
    Used to fuse precomputed e_text with precomputed e_img token sequences.
    """
    def __init__(self, in_channels: int = 768, out_channels: int = 1024):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    def convert_to_fp16(self):
        self.half()
        return self

    def convert_to_fp32(self):
        self.float()
        return self

