"""
Ref:https://nn.labml.ai/diffusion/stable_diffusion/model/unet.html
https://github.com/explainingai-code/DDPM-Pytorch/blob/main/models/unet_base.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding module.
    This module computes positional encodings for given time steps.
    Taken from Positional Embedding used in the original transformer paper "Attention is All You Need".
    """
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000.0) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """
    Residual block module.
    Group normalization and SiLU activation are used.
    Additionally, a time embedding can be added to the input (for diffusion models).
    
    Args:
    - in_channels (int): The number of input channels.
    - out_channels (int): The number of output channels.
    - time_emb_dim (int, optional): The time embedding dimension. Defaults to None.
    """


    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

        self.time_emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.time_emb_dim, out_channels))

        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.in_layers(x)

        # Apply time embedding if provided
        if time_emb is not None:
            time_emb = self.time_emb_layer(time_emb).unsqueeze(2).unsqueeze(3)
            h = h + time_emb

        h = self.out_layers(h)
        skip = self.skip(x)

        return h + skip


class UNet(nn.Module):
    """
    UNet model for image denoising.

    This module implements the UNet architecture for image denoising.
    Consists of Down, Middle, and Up blocks built with Residual and Attention blocks.

    
    Args:
    - in_channels (int): The number of input channels.
    - down_channels (tuple): The number of channels in each down block.
    - up_channels (tuple): The number of channels in each up block.
    - time_emb_dim (int, optional): The time embedding dimension. Defaults to 128.
    - num_attention_heads (int, optional): The number of attention heads. Defaults to 2.
    """
    def __init__(self,
                 in_channels=3,
                 down_channels=(64, 128, 256, 512),
                 up_channels=(512, 256, 128, 64),
                 time_emb_dim=128,
                 num_attention_heads=2):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.positional_embedding = PositionalEmbedding(time_emb_dim)

        self.conv0 = nn.Conv2d(self.in_channels, down_channels[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList([])
        for i in range(len(down_channels)):
            in_channels = down_channels[i - 1] if i > 0 else down_channels[i]
            out_channels = down_channels[i]

            self.down_blocks.append(nn.ModuleList([
                ResBlock(in_channels, out_channels, time_emb_dim),
                ResBlock(out_channels, out_channels, time_emb_dim),
                nn.MultiheadAttention(out_channels, num_heads=num_attention_heads),
                DownBlock(out_channels, out_channels)]))

        self.middle_block = nn.ModuleList([
            ResBlock(down_channels[-1], down_channels[-1], time_emb_dim),
            nn.MultiheadAttention(down_channels[-1], num_heads=num_attention_heads),
            ResBlock(down_channels[-1], down_channels[-1], time_emb_dim)])

        self.up_blocks = nn.ModuleList([])
        for i in range(len(up_channels)):
            in_channels = up_channels[i]
            out_channels = up_channels[i + 1] if i < len(up_channels) - 1 else in_channels

            self.up_blocks.append(nn.ModuleList([
                ResBlock(in_channels, out_channels, time_emb_dim),
                ResBlock(out_channels, out_channels, time_emb_dim),
                nn.MultiheadAttention(out_channels, num_heads=num_attention_heads),
                UpBlock(out_channels, out_channels)]))

        self.last_conv = nn.Conv2d(up_channels[-1], self.in_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        # Positional embedding
        time_emb = self.positional_embedding(time)

        x = self.conv0(x)

        down_feats = []
        for down_block in self.down_blocks:
            block1, block2, attn_block, down_block = down_block
            x = block1(x, time_emb)
            x = block2(x, time_emb)

            # Attention block
            b, c, h, w = x.shape
            attn = x.reshape(b, c, h * w)
            attn = F.group_norm(attn, 32)
            attn = attn.permute(0, 2, 1)
            attn = attn_block(attn, attn, attn)[0]
            attn = attn.permute(0, 2, 1).reshape(b, c, h, w)
            x = attn + x

            x = down_block(x)
            down_feats.append(x)

        x = self.middle_block[0](x, time_emb)

        # Attention block
        b, c, h, w = x.shape
        attn = x.reshape(b, c, h * w)
        attn = F.group_norm(attn, 32)
        attn = attn.permute(0, 2, 1)
        attn = self.middle_block[1](attn, attn, attn)[0]
        attn = attn.permute(0, 2, 1).reshape(b, c, h, w)
        x = attn + x

        x = self.middle_block[2](x)

        for up_block in self.up_blocks:
            block1, block2, attn_block, up_block = up_block

            x = x + down_feats.pop()

            x = block1(x, time_emb)
            x = block2(x, time_emb)

            # Attention block
            b, c, h, w = x.shape
            attn = x.reshape(b, c, h * w)
            attn = F.group_norm(attn, 32)
            attn = attn.permute(0, 2, 1)
            attn = attn_block(attn, attn, attn)[0]
            attn = attn.permute(0, 2, 1).reshape(b, c, h, w)
            x = attn + x

            x = up_block(x)

        x = self.last_conv(x)
        return x


class DownBlock(nn.Module):
    """
    Downsampling block module.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super(DownBlock, self).__init__()
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downsample(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block module.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


# Example usage
if __name__ == "__main__":
    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    time = torch.randint(1, 128, (x.shape[0],))  # Example time embedding
    output = model(x, time)
    print(output.shape)  # Should be (1, 3, 256, 256)