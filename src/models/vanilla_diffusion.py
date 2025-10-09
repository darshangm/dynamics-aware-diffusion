import torch
from .base_diffusion import BaseDiffusionModel, DiffusionMethod
from ..training.base_trainer import BaseDiffusionTrainer


class VanillaDiffusionModel(BaseDiffusionModel):
    """
    Vanilla diffusion model - identical to your original implementation.
    Uses the base model without any modifications.
    """
    pass


class VanillaDiffusionTrainer(BaseDiffusionTrainer):
    """
    Vanilla diffusion trainer - no constraints applied.
    Uses the base trainer without any modifications.
    """
    pass


class VanillaDiffusion(DiffusionMethod):
    """Vanilla diffusion method implementation."""
    
    def create_model(self, input_dim, condition_dim, seq_len, **kwargs):
        """Create vanilla diffusion model."""
        return VanillaDiffusionModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            seq_len=seq_len,
            time_dim=kwargs.get('time_dim', 128),
            hidden_dim=kwargs.get('hidden_dim', 256),
            latent_dim=kwargs.get('latent_dim', 128)
        )
    
    def create_trainer(self, model, **kwargs):
        """Create vanilla diffusion trainer."""
        return VanillaDiffusionTrainer(
            model=model,
            beta_start=kwargs.get('beta_start', 1e-4),
            beta_end=kwargs.get('beta_end', 0.02),
            timesteps=kwargs.get('timesteps', 800)
        )
    
    def get_method_name(self):
        """Return method name for saving/logging."""
        return "vanilla"


def load_vanilla_model(model_path, input_dim, condition_dim, seq_len=30, device=None):
    """
    Load a trained vanilla diffusion model from checkpoint.
    This function maintains compatibility with your existing saved models.
    
    Args:
        model_path: Path to the saved model checkpoint
        input_dim: Input dimension (state_dim + control_dim)
        condition_dim: Condition dimension (2 * state_dim)
        seq_len: Sequence length
        device: Device to load model on
        
    Returns:
        model: Loaded model
        trainer: Diffusion trainer with the loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with the same architecture as during training
    model = VanillaDiffusionModel(
        input_dim=input_dim,
        condition_dim=condition_dim,
        time_dim=128,
        seq_len=seq_len,
        hidden_dim=256
    ).to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Create trainer with the loaded model
    trainer = VanillaDiffusionTrainer(
        model=model,
        timesteps=800
    )
    
    print(f"Vanilla model loaded from {model_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, trainer