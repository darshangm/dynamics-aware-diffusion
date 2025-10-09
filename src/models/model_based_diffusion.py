import torch
import numpy as np
from .base_diffusion import BaseDiffusionModel, DiffusionMethod
from ..training.base_trainer import BaseDiffusionTrainer
from ..utils.dynamics import construct_dynamics_matrices


class ModelBasedDiffusionModel(BaseDiffusionModel):
    """
    Model-based diffusion model - identical to base model.
    The physics constraints are applied in the trainer, not the model.
    """
    pass


class ModelBasedDiffusionTrainer(BaseDiffusionTrainer):
    """
    Model-based diffusion trainer with physics-based projection.
    """
    
    def __init__(
        self,
        model,
        A_T,  # System dynamics matrix A_T
        C_T,  # System dynamics matrix C_T  
        state_dim,
        control_dim,
        seq_len,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.seq_len = seq_len
        
        # Store dynamics matrices
        self.A_T = A_T
        self.C_T = C_T
        
        # Construct the full dynamics matrix H = [A_T C_T; 0 I]
        self._construct_dynamics_matrix()
    
    def _construct_dynamics_matrix(self):
        """Construct the full dynamics matrix H = [A_T C_T; 0 I]"""
        # Convert numpy matrices to torch tensors if needed
        if isinstance(self.A_T, np.ndarray):
            self.A_T = torch.tensor(self.A_T, dtype=torch.float32)
        if isinstance(self.C_T, np.ndarray):
            self.C_T = torch.tensor(self.C_T, dtype=torch.float32)
        
        # Get dimensions
        state_total_dim = self.seq_len * self.state_dim
        control_total_dim = self.seq_len * self.control_dim
        
        # Create the full dynamics matrix H
        H_top = torch.cat([self.A_T, self.C_T], dim=1)
        
        # Create the identity matrix for controls
        H_bottom_A = torch.zeros(control_total_dim, self.state_dim)
        H_bottom_B = torch.eye(control_total_dim)
        H_bottom = torch.cat([H_bottom_A, H_bottom_B], dim=1)
        
        # Combine top and bottom to form H
        self.H = torch.cat([H_top, H_bottom], dim=0)
        
        # Move to device
        self.H = self.H.to(self.device)
    
    def to_device(self):
        """Move all tensors to device, including dynamics matrices."""
        super().to_device()
        if hasattr(self, 'H'):
            self.H = self.H.to(self.device)
        if hasattr(self, 'A_T'):
            self.A_T = self.A_T.to(self.device)
        if hasattr(self, 'C_T'):
            self.C_T = self.C_T.to(self.device)
    
    def apply_constraints(self, pred_x0, condition, t):
        """
        Apply physics-based constraints using system dynamics.
        
        Args:
            pred_x0: Predicted clean trajectory [batch_size, seq_len, state_dim + control_dim]
            condition: Conditioning information [batch_size, condition_dim]
            t: Current timestep [batch_size]
            
        Returns:
            Constrained trajectory following system dynamics
        """
        if not hasattr(self, 'H'):
            return pred_x0
        
        batch_size = pred_x0.shape[0]
        
        # Extract initial states from condition (first state_dim elements)
        initial_states = condition[:, :self.state_dim]
        
        # Project the trajectory to follow system dynamics
        projected_x = self._dynamics_projection(pred_x0, initial_states)
        
        # Interpolate between the model prediction and projected prediction
        # based on diffusion timestep (stronger projection as t decreases)
        alpha_t_cumprod = self.alphas_cumprod[t].reshape(-1, 1, 1)
        constrained_x = alpha_t_cumprod * projected_x + (1 - alpha_t_cumprod) * pred_x0
        
        return constrained_x
    
    def _dynamics_projection(self, x, initial_states):
        """
        Project the trajectory onto the manifold defined by the system dynamics.
        
        Args:
            x: Batch of trajectories [batch_size, seq_len, state_dim + control_dim]
            initial_states: Initial states [batch_size, state_dim]
        
        Returns:
            Projected trajectories that follow the system dynamics
        """
        batch_size = x.shape[0]
        
        # Reshape the trajectories for matrix operations
        # [batch_size, seq_len, state_dim + control_dim] -> [batch_size, (seq_len)*(state_dim + control_dim)]
        x_flat = x.reshape(batch_size, -1)
        
        # For each trajectory in the batch
        projected_x_flat = []
        
        for i in range(batch_size):
            # Extract initial state for this trajectory
            x0 = initial_states[i]
            
            # Extract control inputs for this trajectory
            # [seq_len * control_dim]
            u_flat = x_flat[i, self.seq_len * self.state_dim:]
            
            # Combine initial state and control inputs
            xu0 = torch.cat([x0, u_flat])
            
            # Project using the dynamics matrix H
            # [x(0:T); u(0:T-1)] = H [x(0); u(0:T-1)]
            proj_x_flat = torch.matmul(self.H, xu0)
            
            projected_x_flat.append(proj_x_flat)
        
        # Stack batch results
        projected_x_flat = torch.stack(projected_x_flat)
        
        # Reshape back to original format
        # [batch_size, (seq_len)*(state_dim + control_dim)] -> [batch_size, seq_len, state_dim + control_dim]
        projected_x = projected_x_flat.reshape(batch_size, self.seq_len, self.state_dim + self.control_dim)
        
        return projected_x


class ModelBasedDiffusion(DiffusionMethod):
    """Model-based diffusion method implementation."""
    
    def create_model(self, input_dim, condition_dim, seq_len, **kwargs):
        """Create model-based diffusion model."""
        return ModelBasedDiffusionModel(
            input_dim=input_dim,
            condition_dim=condition_dim,
            seq_len=seq_len,
            time_dim=kwargs.get('time_dim', 128),
            hidden_dim=kwargs.get('hidden_dim', 256),
            latent_dim=kwargs.get('latent_dim', 128)
        )
    
    def create_trainer(self, model, state_dim, control_dim, seq_len, **kwargs):
        """Create model-based diffusion trainer."""
        # Construct dynamics matrices
        A_T, C_T = construct_dynamics_matrices(state_dim, control_dim, seq_len)
        
        return ModelBasedDiffusionTrainer(
            model=model,
            A_T=A_T,
            C_T=C_T,
            state_dim=state_dim,
            control_dim=control_dim,
            seq_len=seq_len,
            beta_start=kwargs.get('beta_start', 1e-4),
            beta_end=kwargs.get('beta_end', 0.02),
            timesteps=kwargs.get('timesteps', 800)
        )
    
    def get_method_name(self):
        """Return method name for saving/logging."""
        return "model_based"


def load_model_based_model(model_path, input_dim, condition_dim, seq_len=30, 
                          state_dim=4, control_dim=2, device=None):
    """
    Load a trained model-based diffusion model from checkpoint.
    
    Args:
        model_path: Path to the saved model checkpoint
        input_dim: Input dimension (state_dim + control_dim)
        condition_dim: Condition dimension (2 * state_dim)
        seq_len: Sequence length
        state_dim: State dimension
        control_dim: Control dimension
        device: Device to load model on
        
    Returns:
        model: Loaded model
        trainer: Diffusion trainer with the loaded model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with the same architecture as during training
    model = ModelBasedDiffusionModel(
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
    
    # Construct dynamics matrices
    A_T, C_T = construct_dynamics_matrices(state_dim, control_dim, seq_len)
    
    # Create trainer with the loaded model
    trainer = ModelBasedDiffusionTrainer(
        model=model,
        A_T=A_T,
        C_T=C_T,
        state_dim=state_dim,
        control_dim=control_dim,
        seq_len=seq_len,
        timesteps=800
    )
    
    print(f"Model-based model loaded from {model_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    
    return model, trainer