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
    Handles conditioning on initial states and reward guidance.
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 guide_fn: Optional[Callable] = None,
                 guide_weight: float = 1.0):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer for denormalization
            guide_fn: Optional guidance function for reward-weighted sampling
            guide_weight: Weight for guidance signal
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
    
    def get_action(self, observation, **kwargs) -> np.ndarray:
        """
        Get action for a given observation (for evaluation).
        
        Args:
            observation: Current observation (could be dict or array)
        
        Returns:
            Action to take
        """
        # Handle dict observations (PointMaze, goal-based envs)
        if isinstance(observation, dict):
            # For goal-conditioned tasks, concatenate state with goal
            if 'observation' in observation and 'desired_goal' in observation:
                obs_state = observation['observation']
                obs_goal = observation['desired_goal']
                observation = np.concatenate([obs_state, obs_goal])
            elif 'observation' in observation:
                observation = observation['observation']
            elif 'achieved_goal' in observation:
                observation = observation['achieved_goal']
            else:
                # Concatenate all dict values
                observation = np.concatenate([v.flatten() for v in observation.values()])
        
        # Ensure it's a numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Normalize observation
        observation = observation.reshape(1, -1)
        normed_obs = self.normalizer.normalize_observations(observation)
        
        # Create initial condition
        normed_obs_tensor = torch.FloatTensor(normed_obs).to(self.diffusion.betas.device)
        
        # Condition has both obs and action, set action to zeros initially
        initial_condition = torch.zeros(1, self.transition_dim).to(self.diffusion.betas.device)
        initial_condition[:, :self.observation_dim] = normed_obs_tensor
        
        conditions = {0: initial_condition}
        
        # Sample trajectory
        trajectory = self.sample_loop(batch_size=1, conditions=conditions, verbose=False)

        # DEBUG: Check trajectory structure
        trajectory_np = trajectory[0].cpu().numpy()
        # print(f"DEBUG - Full trajectory shape: {trajectory_np.shape}")
        # print(f"DEBUG - Obs part (first step): {trajectory_np[0, :self.observation_dim]}")
        # print(f"DEBUG - Action part (first step): {trajectory_np[0, self.observation_dim:self.observation_dim+self.action_dim]}")
        # print(f"DEBUG - Obs std: {trajectory_np[:, :self.observation_dim].std():.6f}")
        # print(f"DEBUG - Action std: {trajectory_np[:, self.observation_dim:].std():.6f}")
        # print(f"DEBUG - Trajectory variance: {trajectory_np.std():.6f}")
        # print(f"DEBUG - Trajectory range: [{trajectory_np.min():.3f}, {trajectory_np.max():.3f}]")

        # Extract first action
        action_start = self.observation_dim
        action_end = action_start + self.action_dim

        # Get SECOND action (timestep 1), not first (timestep 0 is conditioned)
        normed_action = trajectory_np[1, action_start:action_end]  # Changed from [0, ...] to [1, ...]
        
        # print(f"DEBUG - Normed action: {normed_action}")

        # Unnormalize
        action = self.normalizer.unnormalize_actions(normed_action.reshape(1, -1))
        # print(f"DEBUG - Unnormed action: {action.flatten()}")

        return action.flatten()

class ValueGuidedPolicy(GuidedPolicy):
    """
    Policy guided by a learned value function.
    Used for reward-weighted trajectory sampling.
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 value_model: nn.Module,
                 guide_weight: float = 1.0):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer
            value_model: Value function V(s) or Q(s,a)
            guide_weight: Weight for value guidance
        """
        # Define guide function using value model
        def guide_fn(x, t):
            # Extract observations from trajectory
            obs = x[:, :, :diffusion_model.observation_dim]
            # Return value estimates
            return value_model(obs).sum(dim=1)
        
        super().__init__(diffusion_model, normalizer, guide_fn, guide_weight)
        self.value_model = value_model


class RewardWeightedPolicy(GuidedPolicy):
    """
    Policy that samples trajectories weighted by cumulative reward.
    Uses classifier-free guidance without explicit value function.
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 returns_scale: float = 1000.0,
                 discount: float = 0.99):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer
            returns_scale: Scale factor for returns
            discount: Discount factor
        """
        super().__init__(diffusion_model, normalizer, guide_fn=None, guide_weight=0.0)
        self.returns_scale = returns_scale
        self.discount = discount
    
    @torch.no_grad()
    def sample_with_returns(self,
                           batch_size: int,
                           returns: float,
                           conditions: Optional[Dict[int, torch.Tensor]] = None,
                           verbose: bool = False) -> torch.Tensor:
        """
        Sample trajectories conditioned on target returns.
        
        Args:
            batch_size: Number of samples
            returns: Target cumulative return
            conditions: Initial state conditions
            verbose: Show progress
        
        Returns:
            Sampled trajectories
        """
        # Normalize returns
        normed_returns = returns / self.returns_scale
        
        # Add returns to conditions if needed
        # In practice, this requires the model to be trained with returns conditioning
        # For now, we sample normally and can filter by estimated returns
        
        trajectories = self.sample_loop(batch_size, conditions, verbose)
        
        return trajectories


class DynamicsAwarePolicy(GuidedPolicy):
    """
    Dynamics-aware diffusion policy using projection-based sampling.
    
    Handles conversion between interleaved format (used by diffusion model)
    and concatenated format (required by projection matrix).
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 projection_matrix: torch.Tensor,
                 state_dim: int,
                 action_dim: int):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer for denormalization
            projection_matrix: P = FF† matrix for concatenated format
            state_dim: State dimension (from dynamics, not observation_dim)
            action_dim: Action dimension
        """
        super().__init__(diffusion_model, normalizer)
        
        self.P = projection_matrix.to(diffusion_model.betas.device)
        self.state_dim = state_dim
        self.action_dim_actual = action_dim
        
        # Note: observation_dim might include goal (e.g., for PointMaze)
        # but state_dim is just the actual state for dynamics
        
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
        
        Note: We need (T+1) states but only T actions.
        """
        batch_size, horizon, transition_dim = trajectory.shape
        
        # Extract observations and actions from interleaved format
        observations = trajectory[:, :, :self.observation_dim]  # (batch, horizon, obs_dim)
        actions = trajectory[:, :, self.observation_dim:]       # (batch, horizon, action_dim)
        
        # For dynamics projection, we need the actual state, not full observation
        # Assuming state is first state_dim elements of observation
        states = observations[:, :, :self.state_dim]  # (batch, horizon, state_dim)
        
        # We need T+1 states: [s(0), s(1), ..., s(T)]
        # But we only have T states in trajectory (horizon steps)
        # We'll predict s(T+1) using last available state + action
        # For now, duplicate last state as approximation
        last_state = states[:, -1:, :]  # (batch, 1, state_dim)
        states_extended = torch.cat([states, last_state], dim=1)  # (batch, horizon+1, state_dim)
        
        # Flatten: concatenate all states, then all actions
        states_flat = states_extended.reshape(batch_size, -1)  # (batch, (T+1)*state_dim)
        actions_flat = actions.reshape(batch_size, -1)         # (batch, T*action_dim)
        
        # Concatenated format
        concatenated = torch.cat([states_flat, actions_flat], dim=1)
        return concatenated
    
    def concatenated_to_interleaved(self, 
                                    concatenated: torch.Tensor, 
                                    horizon: int) -> torch.Tensor:
        """
        Convert from concatenated format back to interleaved format.
        
        Input format (from projection):
            [obs(0), obs(1), ..., obs(T), action(0), action(1), ..., action(T-1)]
            Shape: (batch, (T+1)*state_dim + T*action_dim)
        
        Output format (for diffusion model):
            trajectory[t] = [obs(t), action(t)]
            Shape: (batch, horizon, obs_dim + action_dim)
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
        states = states[:, :-1, :]  # (batch, horizon, state_dim)
        
        # Pad state to observation_dim if needed (e.g., add goal back)
        if self.state_dim < self.observation_dim:
            # Pad with zeros (goal will be set by conditioning later)
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
    def p_sample_with_projection(self,
                                 x: torch.Tensor,
                                 t: torch.Tensor,
                                 i: int,
                                 conditions: Optional[Dict[int, torch.Tensor]] = None) -> torch.Tensor:
        """
        Sample with dynamics-aware projection (Algorithm 1, steps 3-4).
        
        Handles format conversion between interleaved (diffusion) and concatenated (projection).
        """
        batch_size = x.shape[0]
        
        # Step 3: Predict using neural network
        model_mean, model_log_variance = self.diffusion.p_mean_variance(x, t)
        
        # Sample noise
        noise = torch.randn_like(x)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        # τ̂_{i-1} = μ_θ(τ'_i, i) + √β_i ε_i
        tau_hat = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
        # Apply initial conditions before projection (in interleaved format)
        if conditions is not None:
            tau_hat = self.apply_conditions(tau_hat, conditions)
        
        # Step 4: Project onto feasible trajectory space
        if i > 0:
            # Get noise schedule parameters
            beta_i = self.diffusion.betas[i]
            beta_prev = self.diffusion.betas[i - 1] if i > 0 else torch.tensor(0.0).to(x.device)
            
            # Projection weights from paper: √(1-β_{i-1}) and √β_{i-1}
            proj_weight = torch.sqrt(1 - beta_prev)
            noise_weight = torch.sqrt(beta_prev)
            
            # Convert to concatenated format for projection
            tau_hat_concat = self.interleaved_to_concatenated(tau_hat)
            
            # Apply projection: τ'_{i-1} = (√(1-β_{i-1}) P + √β_{i-1} I) τ̂_{i-1}
            tau_proj_concat = proj_weight * (self.P @ tau_hat_concat.T).T + noise_weight * tau_hat_concat
            
            # Convert back to interleaved format
            tau_prime = self.concatenated_to_interleaved(tau_proj_concat, self.horizon)
        else:
            # Final step (i=0): Full projection (β_0 = 0)
            tau_hat_concat = self.interleaved_to_concatenated(tau_hat)
            tau_proj_concat = (self.P @ tau_hat_concat.T).T
            tau_prime = self.concatenated_to_interleaved(tau_proj_concat, self.horizon)
        
        # Reapply conditions after projection (in interleaved format)
        if conditions is not None:
            tau_prime = self.apply_conditions(tau_prime, conditions)
        
        return tau_prime
    
    @torch.no_grad()
    def sample_loop(self,
                   batch_size: int = 1,
                   conditions: Optional[Dict[int, torch.Tensor]] = None,
                   verbose: bool = False) -> torch.Tensor:
        """
        Dynamics-aware sampling loop (Algorithm 1).
        
        Args:
            batch_size: Number of trajectories to sample
            conditions: Optional conditioning (e.g., {0: initial_state})
            verbose: Show progress bar
        
        Returns:
            Sampled feasible trajectories (batch, horizon, transition_dim)
        """
        device = self.diffusion.betas.device
        shape = (batch_size, self.horizon, self.transition_dim)
        
        # Start from noise
        tau_prime = torch.randn(shape, device=device)
        
        # Apply initial conditions
        if conditions is not None:
            tau_prime = self.apply_conditions(tau_prime, conditions)
        
        # Reverse diffusion with projection
        timesteps = list(reversed(range(self.diffusion.n_timesteps)))
        
        if verbose:
            timesteps = tqdm(timesteps, desc='Dynamics-aware Planning', total=self.diffusion.n_timesteps)
        
        for i in timesteps:
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            tau_prime = self.p_sample_with_projection(tau_prime, t, i, conditions)
        
        return tau_prime
    
    def get_action(self, observation, **kwargs) -> np.ndarray:
        """
        Get action using dynamics-aware sampling.
        
        Args:
            observation: Current observation (could be dict or array)
        
        Returns:
            Action to take
        """
        # Handle dict observations (PointMaze, goal-based envs)
        if isinstance(observation, dict):
            # For goal-conditioned tasks, concatenate state with goal
            if 'observation' in observation and 'desired_goal' in observation:
                obs_state = observation['observation']
                obs_goal = observation['desired_goal']
                observation = np.concatenate([obs_state, obs_goal])
            elif 'observation' in observation:
                observation = observation['observation']
            elif 'achieved_goal' in observation:
                observation = observation['achieved_goal']
            else:
                # Concatenate all dict values
                observation = np.concatenate([v.flatten() for v in observation.values()])
        
        # Ensure it's a numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Normalize observation
        observation = observation.reshape(1, -1)
        normed_obs = self.normalizer.normalize_observations(observation)
        
        # Create initial condition
        normed_obs_tensor = torch.FloatTensor(normed_obs).to(self.diffusion.betas.device)
        
        # Condition has both obs and action, set action to zeros initially
        initial_condition = torch.zeros(1, self.transition_dim).to(self.diffusion.betas.device)
        initial_condition[:, :self.observation_dim] = normed_obs_tensor
        
        conditions = {0: initial_condition}
        
        # Sample trajectory with dynamics awareness
        trajectory = self.sample_loop(batch_size=1, conditions=conditions, verbose=False)
        
        # Extract action
        trajectory_np = trajectory[0].cpu().numpy()
        action_start = self.observation_dim
        action_end = action_start + self.action_dim
        
        # Get SECOND action (timestep 1), not first (timestep 0 is conditioned)
        normed_action = trajectory_np[1, action_start:action_end]
        
        # Unnormalize
        action = self.normalizer.unnormalize_actions(normed_action.reshape(1, -1))
        
        return action.flatten()
    
    def update_projection_matrix(self, new_P: torch.Tensor):
        """
        Update projection matrix (useful for different horizons or linearization points).
        
        Args:
            new_P: New projection matrix
        """
        self.P = new_P.to(self.diffusion.betas.device)


class MPCPolicy(GuidedPolicy):
    """
    Model Predictive Control policy using diffusion planning.
    Re-plans at every timestep (more compute, better performance).
    """
    
    def __init__(self,
                 diffusion_model,
                 normalizer,
                 action_horizon: int = 8):
        """
        Args:
            diffusion_model: GaussianDiffusion model
            normalizer: DatasetNormalizer
            action_horizon: How many actions to execute before replanning
        """
        super().__init__(diffusion_model, normalizer)
        self.action_horizon = action_horizon
        self.planned_actions = []
    
    def get_action(self, observation, **kwargs) -> np.ndarray:
        """
        Get action for a given observation (for evaluation).
        
        Args:
            observation: Current observation (could be dict or array)
        
        Returns:
            Action to take
        """
        # Handle dict observations (PointMaze, goal-based envs)
        if isinstance(observation, dict):
            if 'observation' in observation:
                observation = observation['observation']
            elif 'achieved_goal' in observation:
                observation = observation['achieved_goal']
            else:
                # Concatenate all dict values
                observation = np.concatenate([v.flatten() for v in observation.values()])
        
        # Ensure it's a numpy array
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        # Normalize observation
        observation = observation.reshape(1, -1)
        normed_obs = self.normalizer.normalize_observations(observation)
        
        # Create initial condition
        normed_obs_tensor = torch.FloatTensor(normed_obs).to(self.diffusion.betas.device)
        
        # Condition has both obs and action, set action to zeros initially
        initial_condition = torch.zeros(1, self.transition_dim).to(self.diffusion.betas.device)
        initial_condition[:, :self.observation_dim] = normed_obs_tensor
        
        conditions = {0: initial_condition}
        
        # Sample trajectory
        trajectory = self.sample_loop(batch_size=1, conditions=conditions, verbose=False)
        
        # Extract first action
        trajectory_np = trajectory[0].cpu().numpy()
        action_start = self.observation_dim
        action_end = action_start + self.action_dim
        
        # Get first action from trajectory
        normed_action = trajectory_np[0, action_start:action_end]
        
        # Unnormalize
        action = self.normalizer.unnormalize_actions(normed_action.reshape(1, -1))
        
        return action.flatten()

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