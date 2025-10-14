"""
Diffusion model for trajectory planning.
Implements DDPM/DDIM sampling and training objectives.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from tqdm import tqdm

from .temporal_unet import TemporalUnet


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """
    Extract coefficients at specified timesteps and reshape to broadcast.
    
    Args:
        a: Coefficient tensor (n_timesteps,)
        t: Timestep indices (batch_size,)
        x_shape: Shape to broadcast to
    
    Returns:
        Extracted coefficients reshaped for broadcasting
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear schedule from Ho et al. DDPM paper.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Model for trajectory planning.
    
    Implements:
        - Forward diffusion (adding noise)
        - Reverse diffusion (denoising)
        - DDPM and DDIM sampling
        - Training loss computation
    """
    
    def __init__(self,
                 model: TemporalUnet,
                 horizon: int,
                 observation_dim: int,
                 action_dim: int,
                 n_timesteps: int = 1000,
                 loss_type: str = 'l2',
                 clip_denoised: bool = True,
                 predict_epsilon: bool = True,
                 beta_schedule: str = 'cosine'):
        """
        Args:
            model: Temporal U-Net model
            horizon: Planning horizon length
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            n_timesteps: Number of diffusion timesteps
            loss_type: Loss function ('l1' or 'l2')
            clip_denoised: Whether to clip denoised samples to [-1, 1]
            predict_epsilon: Whether to predict noise (True) or x0 (False)
            beta_schedule: Noise schedule ('linear' or 'cosine')
        """
        super().__init__()
        
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        
        # Noise schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', 
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
        # Loss function
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to x_0 to get x_t.
        
        Args:
            x_start: Clean trajectories (batch, horizon, transition_dim)
            t: Timesteps (batch,)
            noise: Optional noise tensor
        
        Returns:
            Noisy trajectories x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance: q(x_{t-1} | x_t, x_0)
        
        Returns:
            (posterior_mean, posterior_log_variance)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance
    
    def p_mean_variance(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for reverse process: p(x_{t-1} | x_t)
        
        Returns:
            (model_mean, posterior_log_variance)
        """
        # Predict noise or x_0
        model_output = self.model(x, t)
        
        if self.predict_epsilon:
            # Model predicts noise
            x_recon = self.predict_start_from_noise(x, t, model_output)
        else:
            # Model predicts x_0 directly
            x_recon = model_output
        
        if self.clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        model_mean, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t)
        
        Args:
            x: Current sample x_t
            t: Current timestep
        
        Returns:
            Previous sample x_{t-1}
        """
        model_mean, model_log_variance = self.p_mean_variance(x, t)
        noise = torch.randn_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, verbose: bool = False) -> torch.Tensor:
        """
        Generate samples using DDPM sampling (full reverse diffusion).
        
        Args:
            shape: Shape of samples to generate (batch, horizon, transition_dim)
            verbose: Whether to show progress bar
        
        Returns:
            Generated samples
        """
        device = self.betas.device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        iterator = reversed(range(self.n_timesteps))
        if verbose:
            iterator = tqdm(iterator, desc='Sampling', total=self.n_timesteps)
        
        for i in iterator:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        return x
    
    def loss(self, x_start: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute diffusion training loss.
        
        Args:
            x_start: Clean trajectories (batch, horizon, transition_dim)
            weights: Optional loss weights
        
        Returns:
            Scalar loss
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x_start.device).long()
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise or x_0
        model_output = self.model(x_noisy, t)
        
        # Compute loss
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start
        
        loss = self.loss_fn(model_output, target)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        return loss.mean()
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Training forward pass."""
        return self.loss(x, *args, **kwargs)


def test_diffusion():
    """Test the diffusion model."""
    batch_size = 4
    horizon = 32
    obs_dim = 17
    action_dim = 6
    transition_dim = obs_dim + action_dim
    
    # Create model
    unet = TemporalUnet(
        transition_dim=transition_dim,
        dim=64,
        dim_mults=(1, 2, 4)
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        horizon=horizon,
        observation_dim=obs_dim,
        action_dim=action_dim,
        n_timesteps=100
    )
    
    # Test training
    x = torch.randn(batch_size, horizon, transition_dim)
    loss = diffusion.loss(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling
    samples = diffusion.p_sample_loop((2, horizon, transition_dim), verbose=True)
    print(f"Sample shape: {samples.shape}")
    
    print("✓ Diffusion model test passed!")


if __name__ == "__main__":
    test_diffusion()