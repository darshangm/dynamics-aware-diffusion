"""
Planning policies using guided diffusion sampling.
Implements conditioning and reward-weighted trajectory optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Dict
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
                observation = np.concatenate([obs_state, obs_goal])
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
        for t in range(1, min(self.action_horizon + 1, self.horizon)):
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
    Dynamics-aware diffusion policy using projection-based sampling.
    Can also use action buffering (MPC-style) via action_horizon parameter.
    
    Handles conversion between interleaved format (used by diffusion model)
    and concatenated format (required by projection matrix).
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 projection_matrix: torch.Tensor,
                 state_dim: int,
                 action_dim: int,
                 action_horizon: Optional[int] = None):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer for denormalization
            projection_matrix: P = FF† matrix for concatenated format
            state_dim: State dimension (from dynamics, not observation_dim)
            action_dim: Action dimension
            action_horizon: How many actions to use before replanning (None = use all)
        """
        # If action_horizon not specified, use full horizon (most efficient)
        if action_horizon is None:
            action_horizon = diffusion_model.horizon
        
        super().__init__(diffusion_model, normalizer, action_horizon=action_horizon)
        
        self.P = projection_matrix.to(diffusion_model.betas.device)
        self.state_dim = state_dim
        self.action_dim_actual = action_dim
        
        print(f"DynamicsAwarePolicy: action_horizon={self.action_horizon}")
        print(f"  Projection matrix shape: {self.P.shape}")
        print(f"  Will replan every {self.action_horizon} steps")
        
        # Validate dimensions
        expected_size = (self.horizon + 1) * state_dim + self.horizon * action_dim
        if self.P.shape[0] != expected_size:
            print(f"Warning: Projection matrix size {self.P.shape[0]} doesn't match "
                  f"expected size {expected_size}")
            print(f"  Horizon: {self.horizon}, State dim: {state_dim}, Action dim: {action_dim}")
    
    def interleaved_to_concatenated(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Convert from interleaved to concatenated format for projection.
        
        Input format (from diffusion model):
            trajectory[t] = [obs(t), action(t)] for t in [0, horizon-1]
            Shape: (batch, horizon, obs_dim + action_dim)
        
        Output format (for projection matrix):
            [obs(0), obs(1), ..., obs(T), action(0), action(1), ..., action(T-1)]
            Shape: (batch, (horizon+1)*state_dim + horizon*action_dim)
        """
        batch_size, horizon, transition_dim = trajectory.shape
        
        # Extract observations and actions from interleaved format
        observations = trajectory[:, :, :self.observation_dim]
        actions = trajectory[:, :, self.observation_dim:]
        
        # For dynamics projection, we need the actual state, not full observation
        states = observations[:, :, :self.state_dim]
        
        # We need T+1 states but only have T states in trajectory
        # Duplicate last state as approximation
        last_state = states[:, -1:, :]
        states_extended = torch.cat([states, last_state], dim=1)
        
        # Flatten: concatenate all states, then all actions
        states_flat = states_extended.reshape(batch_size, -1)
        actions_flat = actions.reshape(batch_size, -1)
        
        # Concatenated format
        concatenated = torch.cat([states_flat, actions_flat], dim=1)
        return concatenated
    
    def concatenated_to_interleaved(self, concatenated: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Convert from concatenated format back to interleaved format.
        
        Input format (from projection):
            [obs(0), obs(1), ..., obs(T), action(0), action(1), ..., action(T-1)]
        
        Output format (for diffusion model):
            trajectory[t] = [obs(t), action(t)]
        """
        batch_size = concatenated.shape[0]
        
        # Split into states and actions
        n_states_total = (horizon + 1) * self.state_dim
        states_flat = concatenated[:, :n_states_total]
        actions_flat = concatenated[:, n_states_total:]
        
        # Reshape
        states = states_flat.reshape(batch_size, horizon + 1, self.state_dim)
        actions = actions_flat.reshape(batch_size, horizon, self.action_dim_actual)
        
        # Drop the last state (we only need T states for T timesteps)
        states = states[:, :-1, :]
        
        # Pad state to observation_dim if needed (e.g., add goal back)
        if self.state_dim < self.observation_dim:
            obs_padding = torch.zeros(
                batch_size, horizon, self.observation_dim - self.state_dim,
                device=states.device
            )
            observations = torch.cat([states, obs_padding], dim=-1)
        else:
            observations = states
        
        # Pad action to action_dim if needed
        if self.action_dim_actual < self.action_dim:
            action_padding = torch.zeros(
                batch_size, horizon, self.action_dim - self.action_dim_actual,
                device=actions.device
            )
            actions = torch.cat([actions, action_padding], dim=-1)
        
        # Interleave: [obs, action] at each timestep
        trajectory = torch.cat([observations, actions], dim=-1)
        
        return trajectory
    
    @torch.no_grad()
    def p_sample_with_projection(self, x: torch.Tensor, t: torch.Tensor, i: int,
                                 conditions: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Sample with dynamics-aware projection (Algorithm 1, steps 3-4).
        """
        batch_size = x.shape[0]
        
        # Step 3: Predict using neural network
        model_mean, model_log_variance = self.diffusion.p_mean_variance(x, t)
        
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        tau_hat = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
        if conditions is not None:
            tau_hat = self.apply_conditions(tau_hat, conditions)
        
        # Step 4: Project onto feasible trajectory space
        if i > 0:
            beta_prev = self.diffusion.betas[i - 1]
            proj_weight = torch.sqrt(1 - beta_prev)
            noise_weight = torch.sqrt(beta_prev)
            
            tau_hat_concat = self.interleaved_to_concatenated(tau_hat)
            # Optimized: Use @ P.T instead of (P @ .T).T
            tau_proj_concat = proj_weight * torch.mm(tau_hat_concat, self.P.T) + noise_weight * tau_hat_concat
            tau_prime = self.concatenated_to_interleaved(tau_proj_concat, self.horizon)
        else:
            # Final step (i=0): Full projection
            tau_hat_concat = self.interleaved_to_concatenated(tau_hat)
            tau_proj_concat = torch.mm(tau_hat_concat, self.P.T)
            tau_prime = self.concatenated_to_interleaved(tau_proj_concat, self.horizon)
        
        if conditions is not None:
            tau_prime = self.apply_conditions(tau_prime, conditions)
        
        return tau_prime
    
    @torch.no_grad()
    def sample_loop(self, batch_size: int = 1, conditions: Optional[Dict[int, torch.Tensor]] = None,
                   verbose: bool = False) -> torch.Tensor:
        """
        Dynamics-aware sampling loop (Algorithm 1).
        Override sample_loop to use projection-based sampling.
        """
        device = self.diffusion.betas.device
        shape = (batch_size, self.horizon, self.transition_dim)
        
        tau_prime = torch.randn(shape, device=device)
        
        if conditions is not None:
            tau_prime = self.apply_conditions(tau_prime, conditions)
        
        timesteps = list(reversed(range(self.diffusion.n_timesteps)))
        
        if verbose:
            timesteps = tqdm(timesteps, desc='Dynamics-aware Planning')
        
        for i in timesteps:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            tau_prime = self.p_sample_with_projection(tau_prime, t, i, conditions)
        
        return tau_prime


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