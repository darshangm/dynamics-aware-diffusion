import torch
import numpy as np
from .base_diffusion import BaseDiffusionModel, DiffusionMethod
from ..training.base_trainer import BaseDiffusionTrainer
from ..utils.hankel_matrix import construct_hankel_matrix


class HankelDiffusionModel(BaseDiffusionModel):
    """
    Hankel diffusion model - identical to base model.
    The Hankel constraints are applied in the trainer, not the model.
    """
    pass


class HankelDiffusionTrainer(BaseDiffusionTrainer):
    """
    Hankel diffusion trainer with Hankel matrix-based projection.
    """
    
    def __init__(
        self,
        model,
        hankel_proj_matrix,
        seq_len,
        feature_dim,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        
        # Store Hankel projection matrix
        self.hankel_proj_matrix = hankel_proj_matrix
        
        # Verify projection matrix dimensions
        expected_dim = seq_len * feature_dim
        if hankel_proj_matrix.shape[0] != expected_dim or hankel_proj_matrix.shape[1] != expected_dim:
            raise ValueError(f"Hankel projection matrix has shape {hankel_proj_matrix.shape}, "
                           f"expected [{expected_dim}, {expected_dim}]")
    
    def to_device(self):
        """Move all tensors to device, including Hankel projection matrix."""
        super().to_device()
        if hasattr(self, 'hankel_proj_matrix'):
            self.hankel_proj_matrix = self.hankel_proj_matrix.to(self.device)
    
    def apply_constraints(self, pred_x0, condition, t):
        """
        Apply Hankel-based constraints using data-driven projection.
        
        Args:
            pred_x0: Predicted clean trajectory [batch_size, seq_len, feature_dim]
            condition: Conditioning information [batch_size, condition_dim]
            t: Current timestep [batch_size]
            
        Returns:
            Constrained trajectory following Hankel projection
        """
        if self.hankel_proj_matrix is None:
            return pred_x0
        
        try:
            # Project the predicted trajectory
            projected_pred_x0 = self._hankel_projection(pred_x0)
            
            # Interpolate between the model prediction and projected prediction
            # based on diffusion timestep (stronger projection as t decreases)
            alpha_t_cumprod = self.alphas_cumprod[t].reshape(-1, 1, 1)
            constrained_x = alpha_t_cumprod * projected_pred_x0 + (1 - alpha_t_cumprod) * pred_x0
            
            return constrained_x
            
        except ValueError as e:
            print(f"Warning: Hankel projection failed: {e}")
            # Continue without projection if it fails
            return pred_x0
    
    def _hankel_projection(self, x):
        """
        Project trajectories onto the column space of the Hankel matrix.
        
        Args:
            x: Batch of trajectories [batch_size, seq_len, feature_dim]
            
        Returns:
            Projected trajectories with the same shape
        """
        batch_size = x.shape[0]
        
        # Verify input dimensions
        if x.shape[1] != self.seq_len or x.shape[2] != self.feature_dim:
            raise ValueError(f"Trajectory shape mismatch. Expected [{batch_size}, {self.seq_len}, {self.feature_dim}], "
                           f"got {x.shape}")
        
        # Reshape to match Hankel column format
        # [batch_size, seq_len, feature_dim] -> [batch_size, seq_len*feature_dim]
        x_flat = x.reshape(batch_size, -1)
        
        # Apply projection: P * x
        # proj_matrix shape: [seq_len*feature_dim, seq_len*feature_dim]
        # x_flat shape: [batch_size, seq_len*feature_dim]
        x_proj_flat = torch.matmul(self.hankel_proj_matrix, x_flat.t()).t()
        
        # Reshape back to original format
        x_proj = x_proj_flat.reshape(batch_size, self.seq_len, self.feature_dim)
        
        return x_proj


class HankelDiffusion(DiffusionMethod):
    """Hankel diffusion method implementation."""
    
    def create_model(self, input_dim, condition_dim, seq_len, **kwargs):
        """Create Hankel diffusion model."""
        return HankelDiffusionModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            seq_len=seq_len,
            time_dim=kwargs.get('time_dim', 128),
            hidden_dim=kwargs.get('hidden_dim', 256),
            latent_dim=kwargs.get('latent_dim', 128)
        )
    
    def create_trainer(self, model, hankel_data_dir, state_dim, control_dim, seq_len, **kwargs):
        """Create Hankel diffusion trainer."""
        # Construct Hankel matrix and projection
        _, proj_matrix, feature_dim = construct_hankel_matrix(
            data_dir=hankel_data_dir,
            state_dim=state_dim,
            control_dim=control_dim,
            seq_len=seq_len,
            max_trajectories=kwargs.get('max_trajectories', 500)
        )
        
        return HankelDiffusionTrainer(
            model=model,
            hankel_proj_matrix=proj_matrix,
            seq_len=seq_len,
            feature_dim=feature_dim,
            beta_start=kwargs.get('beta_start', 1e-4),
            beta_end=kwargs.get('beta_end', 0.02),
            timesteps=kwargs.get('timesteps', 800)
        )
    
    def get_method_name(self):
        """Return method name for saving/logging."""
        return "hankel"


def load_hankel_model(model_path, input_dim, condition_dim, seq_len=30,
                     hankel_data_dir=None, state_dim=4, control_dim=2, device=None):
    """
    Load a trained Hankel diffusion model from checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        input_dim: Input dimension (state_dim + control_dim)
        condition_dim: Condition dimension (2 * state_dim)
        seq_len: Sequence length
        hankel_data_dir: Directory with Hankel matrix data (required for reconstruction)
        state_dim: State dimension
        control_dim: Control dimension
        device: Device to load model on
        
    Returns:
        model: Loaded model
        trainer: Diffusion trainer with the loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if hankel_data_dir is None:
        raise ValueError("hankel_data_dir must be provided to reconstruct Hankel projection matrix")
    
    # Initialize model with the same architecture as during training
    model = HankelDiffusionModel(
        input_dim=input_dim,
        condition_dim=condition_dim,
        time_dim=128,
        seq_len=seq_len,
        hidden_dim=256
    ).to(device)
    
    # Load saved weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    # Reconstruct Hankel matrix and projection
    _, proj_matrix, feature_dim = construct_hankel_matrix(
        data_dir=hankel_data_dir,
        state_dim=state_dim,
        control_dim=control_dim,
        seq_len=seq_len,
        max_trajectories=500
    )
    
    # Create trainer with the loaded model
    trainer = HankelDiffusionTrainer(
        model=model,
        hankel_proj_matrix=proj_matrix,
        seq_len=seq_len,
        feature_dim=feature_dim,
        timesteps=800
    )
    
    print(f"Hankel model loaded from {model_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, trainer