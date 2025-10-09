import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class TimeEmbedding(nn.Module):
    """Time embedding layer for diffusion timesteps."""
    
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(1, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, dim)
        )

    def forward(self, time):
        # Normalize time to [0, 1] range for better stability
        t_norm = time.float() / 1000.0  
        t_embed = self.embedding(t_norm.unsqueeze(-1))
        return t_embed


class TrajectoryEncoder(nn.Module):
    """Encodes an entire trajectory into a latent representation."""
    
    def __init__(self, seq_len, feature_dim, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        # 1D Convolutional layers to process sequence data
        self.conv_layers = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # Global pooling to get sequence-level features
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection
        self.projection = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, feature_dim]
        batch_size = x.shape[0]
        
        # Reshape for 1D convolution: [batch_size, feature_dim, seq_len]
        x = x.permute(0, 2, 1)
        
        # Apply convolutions
        features = self.conv_layers(x)
        
        # Global pooling
        pooled = self.global_pool(features).squeeze(-1)
        
        # Project to latent space
        latent = self.projection(pooled)
        
        return latent


class TrajectoryDecoder(nn.Module):
    """Decodes a latent representation back into a full trajectory."""
    
    def __init__(self, seq_len, feature_dim, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        # Initial projection
        self.initial_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Upsample to sequence length
        self.upsample = nn.Linear(hidden_dim, seq_len * hidden_dim)
        
        # 1D Transposed convolutions
        self.deconv_layers = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, feature_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, latent):
        # latent shape: [batch_size, latent_dim]
        batch_size = latent.shape[0]
        
        # Initial projection
        hidden = self.initial_proj(latent)
        
        # Upsample to sequence length
        upsampled = self.upsample(hidden)
        upsampled = upsampled.view(batch_size, -1, self.seq_len)
        
        # Apply deconvolutions
        trajectory = self.deconv_layers(upsampled)
        
        # Reshape to [batch_size, seq_len, feature_dim]
        trajectory = trajectory.permute(0, 2, 1)
        
        return trajectory


class BaseDiffusionModel(nn.Module):
    """
    Base diffusion model that is identical to your original implementation.
    This ensures compatibility with existing saved models.
    """
    
    def __init__(
        self,
        input_dim,        # Combined dimension of states and controls
        condition_dim,    # Dimension of initial and target states
        seq_len,          # Length of trajectories
        time_dim=128,     # Dimension of time embedding
        hidden_dim=256,   # Hidden dimension
        latent_dim=128    # Latent dimension for trajectory encoding
    ):
        super().__init__()
        
        # Store dimensions
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.seq_len = seq_len
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        # Condition embedding
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, time_dim),
            nn.SiLU()
        )
        
        # Trajectory encoder
        self.encoder = TrajectoryEncoder(
            seq_len=seq_len, 
            feature_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        # Processing network for latent trajectory representation
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim + time_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            seq_len=seq_len,
            feature_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

    def forward(self, x, time, condition):
        """
        Process entire trajectories holistically.
        
        Args:
            x: Input noisy trajectory [batch_size, sequence_length, feature_dim]
            time: Diffusion timestep tokens [batch_size]
            condition: Conditioning information [batch_size, condition_dim]
        """
        batch_size = x.shape[0]
        
        # Get time embedding
        t_embed = self.time_embed(time)  # [batch_size, time_dim]
        
        # Get condition embedding
        cond_embed = self.condition_embed(condition)  # [batch_size, time_dim]
        
        # Encode trajectory into latent space
        trajectory_latent = self.encoder(x)  # [batch_size, latent_dim]
        
        # Concatenate latent with time and condition embeddings
        combined_latent = torch.cat([trajectory_latent, t_embed, cond_embed], dim=1)
        
        # Process the latent representation
        processed_latent = self.latent_processor(combined_latent)
        
        # Decode back to full trajectory
        output_trajectory = self.decoder(processed_latent)
        
        return output_trajectory


class DiffusionMethod(ABC):
    """
    Abstract base class for different diffusion methods.
    Each specific method (vanilla, model-based, hankel) inherits from this.
    """
    
    @abstractmethod
    def create_model(self, input_dim, condition_dim, seq_len, **kwargs):
        """Create the diffusion model for this method."""
        pass
    
    @abstractmethod
    def create_trainer(self, model, **kwargs):
        """Create the trainer for this method."""
        pass
    
    @abstractmethod
    def get_method_name(self):
        """Return the method name for saving/logging."""
        pass
    
    def get_model_save_name(self, epoch):
        """Generate standardized model save name."""
        return f"{self.get_method_name()}_diffusion_model_epoch_{epoch}.pt"