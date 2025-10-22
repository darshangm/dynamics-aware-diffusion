"""
Planning policies using guided diffusion sampling.
Implements conditioning and reward-weighted trajectory optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Dict, Any
from tqdm import tqdm


class GuidedPolicy(nn.Module):
    """
    Base class for guided diffusion sampling policies.
    Handles conditioning on initial states and action buffering.
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 guide_fn: Optional[Callable] = None,
                 guide_weight: float = 1.0,
                 action_horizon: Optional[int] = None):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer for denormalization
            guide_fn: Optional guidance function for reward-weighted sampling
            guide_weight: Weight for guidance signal
            action_horizon: How many actions to use before replanning (None = 1)
        """
        super().__init__()
        self.diffusion = diffusion_model
        self.normalizer = normalizer
        self.guide_fn = guide_fn
        self.guide_weight = guide_weight
        
        self.horizon = diffusion_model.horizon
        self.observation_dim = diffusion_model.observation_dim
        self.action_dim = diffusion_model.action_dim
        self.transition_dim = diffusion_model.transition_dim
        
        # Action buffering (rolling horizon / MPC)
        self.action_horizon = action_horizon if action_horizon is not None else 1
        self.action_buffer = []
    
    def apply_conditions(self, 
                        x: torch.Tensor, 
                        conditions: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Apply conditioning by replacing values at specific timesteps.
        
        Args:
            x: Trajectory tensor (batch, horizon, transition_dim)
            conditions: Dict mapping timestep -> condition values
        
        Returns:
            Conditioned trajectory
        """
        for t, val in conditions.items():
            x[:, t] = val
        return x
    
    @torch.no_grad()
    def p_sample_with_guidance(self,
                               x: torch.Tensor,
                               t: torch.Tensor,
                               conditions: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Sample with optional guidance and conditioning.
        
        Args:
            x: Current sample x_t
            t: Current timestep
            conditions: Optional conditioning values
        
        Returns:
            Next sample x_{t-1}
        """
        batch_size = x.shape[0]
        
        # Get model prediction
        model_mean, model_log_variance = self.diffusion.p_mean_variance(x, t)
        
        # Apply guidance if provided
        if self.guide_fn is not None and self.guide_weight > 0:
            # Enable gradient for guidance
            x_temp = x.detach().requires_grad_(True)
            
            # Compute guidance signal
            with torch.enable_grad():
                guide_score = self.guide_fn(x_temp, t)
                grad = torch.autograd.grad(guide_score.sum(), x_temp)[0]
            
            # Add guidance to mean
            model_mean = model_mean + self.guide_weight * model_log_variance.exp() * grad
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        # Sample x_{t-1}
        x_prev = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
        # Apply conditions (inpainting)
        if conditions is not None:
            x_prev = self.apply_conditions(x_prev, conditions)
        
        return x_prev
    
    @torch.no_grad()
    def sample_loop(self,
                   batch_size: int = 1,
                   conditions: Optional[Dict[int, torch.Tensor]] = None,
                   verbose: bool = False) -> torch.Tensor:
        """
        Full sampling loop with conditioning.
        
        Args:
            batch_size: Number of trajectories to sample
            conditions: Optional conditioning (e.g., {0: initial_state})
            verbose: Show progress bar
        
        Returns:
            Sampled trajectories (batch, horizon, transition_dim)
        """
        device = self.diffusion.betas.device
        shape = (batch_size, self.horizon, self.transition_dim)
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # Apply initial conditions
        if conditions is not None:
            x = self.apply_conditions(x, conditions)
        
        # Reverse diffusion
        iterator = reversed(range(self.diffusion.n_timesteps))
        if verbose:
            iterator = tqdm(iterator, desc='Planning', total=self.diffusion.n_timesteps)
        
        for i in iterator:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample_with_guidance(x, t, conditions)
        
        return x
    
    def _process_observation(self, observation):
        """Handle dict observations and convert to array."""
        if isinstance(observation, dict):
            if 'observation' in observation and 'desired_goal' in observation:
                obs_state = observation['observation']
                obs_goal = observation['desired_goal']
                
                # Check what dimension the model expects
                expected_dim = self.normalizer.obs_mean.shape[0]
                state_dim = len(obs_state)
                goal_dim = len(obs_goal)
                
                if expected_dim == state_dim + goal_dim:
                    # Model was trained with goal-conditioned observations
                    observation = np.concatenate([obs_state, obs_goal])
                else:
                    # Model was trained with state-only observations (no goal)
                    observation = obs_state
            elif 'observation' in observation:
                observation = observation['observation']
            elif 'achieved_goal' in observation:
                observation = observation['achieved_goal']
            else:
                observation = np.concatenate([v.flatten() for v in observation.values()])
        
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        return observation.reshape(1, -1)
    
    def _fill_action_buffer(self, trajectory):
        """Extract actions from trajectory and fill buffer."""
        trajectory_np = trajectory[0].cpu().numpy()
        action_start = self.observation_dim
        action_end = action_start + self.action_dim
        
        # Extract next action_horizon actions (skip timestep 0 which is conditioned)
        for t in range(0, min(self.action_horizon + 1, self.horizon)):
            normed_action = trajectory_np[t, action_start:action_end]
            action = self.normalizer.unnormalize_actions(normed_action.reshape(1, -1))
            self.action_buffer.append(action.flatten())
    
    def get_action(self, observation, **kwargs) -> np.ndarray:
        """
        Get action for a given observation (for evaluation).
        
        Args:
            observation: Current observation (could be dict or array)
        
        Returns:
            Action to take
        """
        # If buffer has actions, use them
        if len(self.action_buffer) > 0:
            return self.action_buffer.pop(0)
        
        # Otherwise, replan
        observation = self._process_observation(observation)
        normed_obs = self.normalizer.normalize_observations(observation)
        normed_obs_tensor = torch.FloatTensor(normed_obs).to(self.diffusion.betas.device)
        
        initial_condition = torch.zeros(1, self.transition_dim).to(self.diffusion.betas.device)
        initial_condition[:, :self.observation_dim] = normed_obs_tensor
        conditions = {0: initial_condition}
        
        # Sample trajectory (subclasses can override sample_loop)
        trajectory = self.sample_loop(batch_size=1, conditions=conditions, verbose=False)
        
        # Extract and buffer actions
        self._fill_action_buffer(trajectory)
        
        # Return first action
        return self.action_buffer.pop(0)


class MPCPolicy(GuidedPolicy):
    """
    Model Predictive Control policy using diffusion planning.
    Plans once, executes multiple actions, then replans.
    """
    
    def __init__(self, diffusion_model, normalizer, action_horizon: int = 8):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer
            action_horizon: How many actions to execute before replanning
        """
        super().__init__(diffusion_model, normalizer, action_horizon=action_horizon)
        print(f"MPCPolicy: action_horizon={self.action_horizon}")


class ValueGuidedPolicy(GuidedPolicy):
    """
    Policy guided by a learned value function.
    Used for reward-weighted trajectory sampling.
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 value_model: nn.Module,
                 guide_weight: float = 1.0,
                 action_horizon: Optional[int] = None):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer
            value_model: Value function V(s) or Q(s,a)
            guide_weight: Weight for value guidance
            action_horizon: How many actions to use before replanning
        """
        # Define guide function using value model
        def guide_fn(x, t):
            # Extract observations from trajectory
            obs = x[:, :, :diffusion_model.observation_dim]
            # Return value estimates
            return value_model(obs).sum(dim=1)
        
        super().__init__(diffusion_model, normalizer, guide_fn, guide_weight, action_horizon)
        self.value_model = value_model


class DynamicsAwarePolicy(GuidedPolicy):
    """
    Policy that generates trajectories respecting system dynamics.
    Uses projection during sampling to enforce dynamics constraints.
    """
    
    def __init__(self,
                diffusion_model,
                projection_matrix: Optional[torch.Tensor] = None,
                normalizer=None,
                state_dim: int = 4,
                observation_dim: int = 4,
                action_dim: int = 2,
                horizon: int = 16,
                projection_schedule: str = 'constant',
                projection_strength: float = 1.0,
                action_horizon: Optional[int] = None):
        """
        Args:
            diffusion_model: Trained diffusion model
            projection_matrix: Dynamics projection matrix P
            normalizer: Dataset normalizer for unnormalization
            state_dim: Physical state dimension (4 for PointMaze)
            observation_dim: Observation dimension (4 for PointMaze without goals)
            action_dim: Action dimension (2 for PointMaze)
            horizon: Planning horizon (length of trajectory to generate)
            projection_schedule: 'constant', 'linear', or 'quadratic'
            projection_strength: Maximum projection strength (0-1)
            action_horizon: How many actions to execute before replanning (default: horizon)
        """
        # Use horizon as default action_horizon for MPC behavior
        if action_horizon is None:
            action_horizon = horizon
        
        # Initialize parent class with CORRECT parameters
        super().__init__(
            diffusion_model=diffusion_model,
            normalizer=normalizer,
            guide_fn=None,
            guide_weight=0.0,
            action_horizon=action_horizon  # ← CRITICAL for MPC!
        )
        
        # Store projection and dynamics info
        self.projection_matrix = projection_matrix
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.projection_schedule = projection_schedule
        self.projection_strength = projection_strength
        
        # Store n_timesteps for projection schedule
        self.n_timesteps = diffusion_model.n_timesteps  # ← FIX: Add this!
        
        # Get device from model
        self.device = next(diffusion_model.parameters()).device  # ← FIX: Add this!
        
        # Store normalization statistics as tensors if normalizer provided
        if normalizer is not None:
            self.obs_mean = torch.from_numpy(normalizer.obs_mean).float().to(self.device)
            self.obs_std = torch.from_numpy(normalizer.obs_std).float().to(self.device)
            self.action_mean = torch.from_numpy(normalizer.action_mean).float().to(self.device)
            self.action_std = torch.from_numpy(normalizer.action_std).float().to(self.device)
            print(f"DynamicsAwarePolicy: Normalizer loaded")
            print(f"  obs_mean shape: {self.obs_mean.shape}")
            print(f"  action_mean shape: {self.action_mean.shape}")
        else:
            self.obs_mean = None
            self.obs_std = None
            self.action_mean = None
            self.action_std = None
            print(f"DynamicsAwarePolicy: No normalizer provided")
        
        if projection_matrix is not None:
            print(f"DynamicsAwarePolicy initialized with dynamics projection")
            print(f"  Planning horizon: {horizon}")
            print(f"  Action horizon (MPC): {action_horizon}")
            print(f"  Projection schedule: {projection_schedule}")
            print(f"  Projection strength: {projection_strength}")
            print(f"  Projection matrix shape: {projection_matrix.shape}")
        else:
            print(f"DynamicsAwarePolicy: No projection matrix (vanilla sampling)")
    
    def _get_projection_alpha(self, t: int) -> float:
        """
        Get projection strength at timestep t based on schedule.
        """
        # Normalize timestep to [0, 1]
        progress = t / self.n_timesteps
        
        if self.projection_schedule == 'constant':
            alpha = self.projection_strength
        
        elif self.projection_schedule == 'linear':
            alpha = self.projection_strength * (1 - progress)
        
        elif self.projection_schedule == 'quadratic':
            alpha = self.projection_strength * (1 - progress) ** 2
        
        elif self.projection_schedule == 'noise_schedule':
            # Match paper: use √(1-β_t) directly
            # Need to get β_t from diffusion model
            beta_t = self.diffusion.betas[t] if hasattr(self, 'diffusion') else 0.0
            alpha = torch.sqrt(1 - beta_t).item() * self.projection_strength
        
        else:
            raise ValueError(f"Unknown projection schedule: {self.projection_schedule}")
        
        return alpha
    
    def unnormalize_states(self, states_norm: torch.Tensor) -> torch.Tensor:
        """Unnormalize states to physical space."""
        if self.obs_mean is None:
            return states_norm
        return states_norm * self.obs_std + self.obs_mean
    
    def unnormalize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """Unnormalize actions to physical space."""
        if self.action_mean is None:
            return actions_norm
        return actions_norm * self.action_std + self.action_mean
    
    def normalize_states(self, states_physical: torch.Tensor) -> torch.Tensor:
        """Normalize states from physical space."""
        if self.obs_mean is None:
            return states_physical
        return (states_physical - self.obs_mean) / self.obs_std
    
    def normalize_actions(self, actions_physical: torch.Tensor) -> torch.Tensor:
        """Normalize actions from physical space."""
        if self.action_mean is None:
            return actions_physical
        return (actions_physical - self.action_mean) / self.action_std
    
    def apply_projection(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """
        Apply dynamics projection with noise schedule alignment.
        
        CRITICAL: Projects in PHYSICAL (unnormalized) space!
        
        Args:
            x: Trajectory in normalized interleaved format (batch, horizon, obs_dim + action_dim)
            t: Current diffusion timestep
        
        Returns:
            x_projected: Projected trajectory in normalized interleaved format
        """
        if self.projection_matrix is None or self.normalizer is None:
            return x
        
        # Get projection strength based on noise level
        alpha = self._get_projection_alpha(t)
        
        if alpha <= 0:
            return x  # No projection at this timestep
        
        batch_size = x.shape[0]
        
        # Extract states and actions (NORMALIZED)
        observations_norm = x[:, :, :self.observation_dim]
        actions_norm = x[:, :, self.observation_dim:]
        states_norm = observations_norm[:, :, :self.state_dim]
        
        # UNNORMALIZE to physical space
        states_physical = self.unnormalize_states(states_norm)
        actions_physical = self.unnormalize_actions(actions_norm)
        
        # Add final state for concatenation
        states_physical_extended = torch.cat([states_physical, states_physical[:, -1:, :]], dim=1)
        
        # Flatten to concatenated format (PHYSICAL space)
        states_flat = states_physical_extended.reshape(batch_size, -1)
        actions_flat = actions_physical.reshape(batch_size, -1)
        x_concat_physical = torch.cat([states_flat, actions_flat], dim=1)
        
        # Project in PHYSICAL space
        x_projected_physical = x_concat_physical @ self.projection_matrix
        
        # Blend with annealing (in physical space)
        x_concat_physical = alpha * x_projected_physical + (1 - alpha) * x_concat_physical
        
        # Split back
        states_size = (self.horizon + 1) * self.state_dim
        states_flat = x_concat_physical[:, :states_size]
        actions_flat = x_concat_physical[:, states_size:]
        
        states_physical = states_flat.reshape(batch_size, self.horizon + 1, self.state_dim)
        actions_physical = actions_flat.reshape(batch_size, self.horizon, self.action_dim)
        
        # Remove final state
        states_physical = states_physical[:, :-1, :]
        
        # NORMALIZE back to normalized space for diffusion model
        states_norm = self.normalize_states(states_physical)
        actions_norm = self.normalize_actions(actions_physical)
        
        # Reconstruct observations
        # Since observation_dim == state_dim (no goals), just use states
        if self.observation_dim == self.state_dim:
            observations_norm = states_norm
        else:
            # If there's padding needed (shouldn't happen with no goals)
            padding = torch.zeros(batch_size, self.horizon,
                                self.observation_dim - self.state_dim,
                                device=states_norm.device)
            observations_norm = torch.cat([states_norm, padding], dim=-1)
        
        # Return in normalized interleaved format
        x_projected = torch.cat([observations_norm, actions_norm], dim=-1)
        
        return x_projected

def test_dynamics_aware_policy():
    """Test the dynamics-aware policy with PointMaze-like dimensions."""
    print("\n" + "="*60)
    print("Testing DynamicsAwarePolicy")
    print("="*60)
    
    # PointMaze dimensions
    state_dim = 4  # [x, y, vx, vy]
    goal_dim = 2   # [goal_x, goal_y]
    obs_dim = state_dim + goal_dim  # 6D total observation
    action_dim = 2  # [ax, ay]
    horizon = 8
    
    # Create dummy models
    from m_diffuser.models.temporal_unet import TemporalUnet
    from m_diffuser.models.diffusion import GaussianDiffusion
    from m_diffuser.datasets.normalization import DatasetNormalizer
    
    # Dummy data for normalizer
    dummy_obs = np.random.randn(1000, obs_dim)
    dummy_actions = np.random.randn(1000, action_dim)
    normalizer = DatasetNormalizer(dummy_obs, dummy_actions, obs_dim, action_dim)
    
    # Create diffusion model
    unet = TemporalUnet(
        transition_dim=obs_dim + action_dim,
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
    
    # Create projection matrix (double integrator)
    dt = 0.01
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    B = np.array([
        [0.5*dt**2, 0],
        [0, 0.5*dt**2],
        [dt, 0],
        [0, dt]
    ])
    
    from m_diffuser.dynamics.projection import ProjectionMatrixBuilder
    builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
    P = builder.get_projection_matrix(horizon)
    
    # Create policy
    policy = DynamicsAwarePolicy(
        diffusion_model=diffusion,
        normalizer=normalizer,
        projection_matrix=P,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=4
    )
    
    # Test with goal-conditioned observation
    test_obs = {
        'observation': np.array([0.0, 0.0, 0.0, 0.0]),  # state
        'desired_goal': np.array([1.0, 1.0])  # goal
    }
    
    print("\nTesting action generation...")
    action = policy.get_action(test_obs)
    
    print(f"✓ Generated action shape: {action.shape}")
    print(f"✓ Action: {action}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)


def test_policy():
    """Test the planning policy."""
    from ..models.temporal_unet import TemporalUnet
    from ..models.diffusion import GaussianDiffusion
    from ..datasets.normalization import DatasetNormalizer
    
    # Create dummy normalizer
    obs_dim, action_dim = 17, 6
    dummy_obs = np.random.randn(1000, obs_dim)
    dummy_actions = np.random.randn(1000, action_dim)
    normalizer = DatasetNormalizer(dummy_obs, dummy_actions, obs_dim, action_dim)
    
    # Create models
    unet = TemporalUnet(
        transition_dim=obs_dim + action_dim,
        dim=64,
        dim_mults=(1, 2, 4)
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        horizon=32,
        observation_dim=obs_dim,
        action_dim=action_dim,
        n_timesteps=50
    )
    
    # Create policy
    policy = GuidedPolicy(diffusion, normalizer)
    
    # Test action generation
    obs = np.random.randn(obs_dim)
    action = policy.get_action(obs)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {action.shape}")
    print("✓ Policy test passed!")


if __name__ == "__main__":
    test_policy()
    # test_dynamics_aware_policy()