from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the diffusion model architecture."""
    input_dim: int = 6  # state_dim + control_dim
    condition_dim: int = 8  # 2 * state_dim
    seq_len: int = 30
    time_dim: int = 128
    hidden_dim: int = 256
    latent_dim: int = 128


@dataclass 
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    epochs: int = 100000
    learning_rate: float = 1e-4
    timesteps: int = 800
    beta_start: float = 1e-4
    beta_end: float = 0.02
    log_interval: int = 100
    save_interval: int = 100


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""
    name: str  # experiment name (e.g., "lqr", "complex_task")
    state_dim: int = 4
    control_dim: int = 2
    
    # Data directories
    data_dir: str = "./data"
    save_dir: str = "./results"
    
    # Nested configs
    model_config: Optional[ModelConfig] = None
    training_config: Optional[TrainingConfig] = None
    
    def __post_init__(self):
        """Initialize nested configs with proper dimensions."""
        if self.model_config is None:
            self.model_config = ModelConfig(
                input_dim=self.state_dim + self.control_dim,
                condition_dim=2 * self.state_dim
            )
        if self.training_config is None:
            self.training_config = TrainingConfig()
    
    @property
    def train_data_dir(self):
        return f"{self.data_dir}/{self.name}/train"
    
    @property
    def test_data_dir(self):
        return f"{self.data_dir}/{self.name}/test"
    
    @property
    def hankel_data_dir(self):
        return f"{self.data_dir}/{self.name}/hankel"
    
    def get_save_dir(self, method_name: str):
        return f"{self.save_dir}/{self.name}/{method_name}"


# Specific experiment configurations
@dataclass
class LQRConfig(ExperimentConfig):
    """Configuration for LQR experiments."""
    name: str = "lqr"
    # Use default values for LQR


@dataclass
class ComplexTaskConfig(ExperimentConfig):
    """Configuration for complex task experiments."""
    name: str = "complex_task"
    # Can override default values here if needed for complex tasks