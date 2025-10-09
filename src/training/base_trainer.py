import torch
import torch.nn.functional as F
from tqdm import tqdm


class BaseDiffusionTrainer:
    """
    Base diffusion trainer containing common diffusion logic.
    Method-specific trainers inherit from this and override specific methods.
    """
    
    def __init__(
        self,
        model,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=800,
        device=None
    ):
        self.model = model
        self.timesteps = timesteps
        
        # Determine device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
        
        # Cosine noise schedule (matches your original implementation)
        self.betas = self._cosine_beta_schedule(timesteps, beta_start, beta_end)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Calculations for diffusion
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Move to device
        self.to_device()
    
    def _cosine_beta_schedule(self, timesteps, beta_start, beta_end):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        Matches your original implementation exactly.
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        
        # Clamp the values
        return torch.clamp(betas, beta_start, beta_end)
    
    def to_device(self):
        """Move all tensors to the appropriate device."""
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Identical to your original implementation.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Get coefficients for the specific timestep
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)
        
        # Add noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, x_start, t, condition, noise=None):
        """
        Compute the MSE loss for training.
        Identical to your original implementation.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_x0 = self.model(x_noisy, t, condition)
        
        # Calculate loss directly on the predicted clean trajectory
        return F.mse_loss(predicted_x0, x_start)
    
    @torch.no_grad()
    def p_sample(self, x, t, condition):
        """
        Sample from p(x_{t-1} | x_t).
        Base implementation - can be overridden by method-specific trainers.
        """
        # Get the predicted clean trajectory
        pred_x0 = self.model(x, t, condition)
        
        # Apply method-specific projection if needed
        pred_x0 = self.apply_constraints(pred_x0, condition, t)
        
        # Calculate mean for the previous timestep using the predicted clean trajectory
        alpha_t = self.alphas[t].reshape(-1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1)
        beta_t = self.betas[t].reshape(-1, 1, 1)
        
        # Calculate the coefficient for prediction
        pred_coef = beta_t * torch.sqrt(alpha_cumprod_t) / (1.0 - alpha_cumprod_t)
        
        # Calculate mean using the predicted x0
        mean = torch.sqrt(alpha_t) * (x - pred_coef * (x - pred_x0)) / torch.sqrt(alpha_t)
        
        # Add noise only for t > 0
        noise = torch.randn_like(x) * torch.sqrt(beta_t) * (t > 0).reshape(-1, 1, 1).float()
        
        return mean + noise
    
    def apply_constraints(self, pred_x0, condition, t):
        """
        Apply method-specific constraints to the predicted trajectory.
        Override this in method-specific trainers.
        
        Args:
            pred_x0: Predicted clean trajectory
            condition: Conditioning information
            t: Current timestep
            
        Returns:
            Constrained trajectory
        """
        # Base implementation: no constraints (vanilla diffusion)
        return pred_x0
    
    @torch.no_grad()
    def sample(self, shape, condition):
        """
        Generate samples using reverse diffusion process.
        
        Args:
            shape: Tuple of (sequence_length, feature_dim)
            condition: Conditioning information [batch_size, condition_dim]
        """
        batch_size = condition.shape[0]
        seq_len, feat_dim = shape
        
        # Start with random noise
        x = torch.randn(batch_size, seq_len, feat_dim, device=self.device)
        
        # Iterate through all timesteps
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            time_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            x = self.p_sample(x, time_tensor, condition)
            
        return x
    
    def train_step(self, x_start, condition, optimizer):
        """
        Single training step.
        Identical to your original implementation.
        """
        batch_size = x_start.shape[0]
        
        # Sample random diffusion timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        optimizer.zero_grad()
        loss = self.compute_loss(x_start, t, condition)
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()