"""
Temporal U-Net for diffusion planning.
Adapted from Janner et al. "Planning with Diffusion for Flexible Behavior Synthesis"
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Timesteps tensor of shape (batch_size,)
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class Downsample1d(nn.Module):
    """1D downsampling layer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """1D upsampling layer."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish activation
    """
    
    def __init__(self, 
                 inp_channels: int, 
                 out_channels: int, 
                 kernel_size: int = 3,
                 n_groups: int = 8):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(nn.Module):
    """
    Residual block with temporal convolutions and time conditioning.
    """
    
    def __init__(self, 
                 inp_channels: int, 
                 out_channels: int, 
                 embed_dim: int = 128,
                 kernel_size: int = 5):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
        )
        
        # Residual connection
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, horizon)
            t: Time embeddings (batch, embed_dim)
        Returns:
            Output tensor (batch, out_channels, horizon)
        """
        out = self.blocks[0](x)
        
        # Add time conditioning
        out = out + self.time_mlp(t)[:, :, None]
        
        out = self.blocks[1](out)
        
        # Residual connection
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):
    """
    Temporal U-Net for diffusion-based trajectory planning.
    
    Architecture:
        - Encoder: Downsample trajectory representations
        - Bottleneck: Process at lowest resolution
        - Decoder: Upsample with skip connections
    """
    
    def __init__(self,
                 transition_dim: int,
                 dim: int = 128,
                 dim_mults: tuple = (1, 2, 4, 8),
                 kernel_size: int = 5,
                 time_dim: Optional[int] = None):
        """
        Args:
            transition_dim: Dimension of state-action pairs (obs_dim + action_dim)
            dim: Base channel dimension
            dim_mults: Channel multipliers for each level
            kernel_size: Kernel size for convolutions
            time_dim: Dimension of time embeddings (default: dim)
        """
        super().__init__()
        
        self.transition_dim = transition_dim
        
        # Time embedding
        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Calculate dimensions at each level
        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # Encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, kernel_size=kernel_size),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, kernel_size=kernel_size),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
        
        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, kernel_size=kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, kernel_size=kernel_size)
        
        # Decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, kernel_size=kernel_size),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, kernel_size=kernel_size),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        # Final output layer
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal U-Net.
        
        Args:
            x: Noisy trajectories (batch, horizon, transition_dim)
            time: Diffusion timesteps (batch,)
            
        Returns:
            Denoised trajectories (batch, horizon, transition_dim)
        """
        # Rearrange to (batch, channels, horizon) for conv1d
        x = x.transpose(1, 2)
        
        # Time embeddings
        t = self.time_mlp(time)
        
        # Encoder
        h = []
        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)
        
        # Bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)
        
        # Decoder with skip connections
        for resnet1, resnet2, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        
        # Final output
        x = self.final_conv(x)
        
        # Rearrange back to (batch, horizon, transition_dim)
        x = x.transpose(1, 2)
        
        return x


def test_temporal_unet():
    """Test the temporal U-Net."""
    batch_size = 4
    horizon = 32
    obs_dim = 17
    action_dim = 6
    transition_dim = obs_dim + action_dim
    
    model = TemporalUnet(
        transition_dim=transition_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8)
    )
    
    # Random input
    x = torch.randn(batch_size, horizon, transition_dim)
    time = torch.randint(0, 1000, (batch_size,))
    
    # Forward pass
    out = model(x, time)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "Output shape should match input shape"
    print("✓ Temporal U-Net test passed!")


if __name__ == "__main__":
    test_temporal_unet()