"""
Diffusion Transformer (DiT) implementation based on:
- "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
- https://arxiv.org/abs/2212.09748
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding module for time steps.
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


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding module.
    Converts image into patches and embeds them.
    """
    def __init__(self, img_size=32, patch_size=2, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns: (B, N, D) where N is number of patches and D is embed_dim
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, N)
        x = x.transpose(1, 2)  # (B, N, embed_dim)
        return x


class MLP(nn.Module):
    """
    MLP block for Transformer.
    """
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Self-attention module with skip connection and normalization.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and MLP.
    Includes time embedding injection.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, 
                 drop=0., attn_drop=0., time_emb_dim=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim)
        ) if time_emb_dim is not None else None
        
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim=dim, hidden_dim=mlp_hidden_dim, dropout=drop)
        
    def forward(self, x, time_emb=None):
        # Apply time embedding if provided
        time_emb_out = 0
        if self.time_mlp is not None and time_emb is not None:
            time_emb_out = self.time_mlp(time_emb)[:, None, :]
            
        x = x + self.attn(x)
        x = x + time_emb_out
        x = x + self.mlp(x)
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer model.
    
    Args:
        img_size (int): Input image size
        patch_size (int): Patch size for tokenizing the image
        in_channels (int): Number of input image channels
        hidden_size (int): Transformer hidden dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): MLP hidden dim ratio
        time_emb_dim (int): Dimension of time embedding
    """
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        time_emb_dim=128,
        dropout=0.0,
    ):
        super().__init__()
        
        # Image embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size,
        )
        self.num_patches = self.patch_embed.n_patches
        
        # Time embedding
        self.time_embed = PositionalEmbedding(time_emb_dim)
        
        # Position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=dropout,
                attn_drop=dropout,
                time_emb_dim=time_emb_dim,
            )
            for _ in range(depth)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def unpatchify(self, x):
        """
        Converts patched feature maps back to image format
        """
        p = self.patch_size
        h = w = int(math.sqrt(x.shape[1]))
        
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = rearrange(x, 'b h w p1 p2 c -> b c (h p1) (w p2)')
        return x
        
    def forward(self, x, time):
        """
        Forward pass
        
        Args:
            x (Tensor): Input images [B, C, H, W]
            time (Tensor): Time embedding [B]
            
        Returns:
            Tensor: Predicted noise [B, C, H, W]
        """
        batch_size = x.shape[0]
        
        # Time embedding
        time_emb = self.time_embed(time)
        
        # Tokenize the image into patches
        x = self.patch_embed(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, time_emb)
        
        # Apply output head
        x = self.norm(x)
        x = self.out_proj(x)
        
        # Reshape back to image format
        x = self.unpatchify(x)
        
        return x


# Example usage
if __name__ == "__main__":
    model = DiT(img_size=32, patch_size=2, in_channels=3, hidden_size=768)
    x = torch.randn(1, 3, 32, 32)
    time = torch.randint(1, 128, (x.shape[0],))  # Example time embedding
    output = model(x, time)
    print(output.shape)  # Should be (1, 3, 32, 32) 