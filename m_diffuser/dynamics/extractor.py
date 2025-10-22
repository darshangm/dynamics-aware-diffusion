"""
Dynamics extraction for different environments.
Extracts (A, B) matrices for linear systems.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Optional


class DynamicsExtractor:
    """
    Base class for extracting system dynamics.
    
    Returns (A, B) matrices where:
        x_{t+1} = A*x_t + B*u_t + w_t
    """
    
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.env = gym.make(env_name)
        
        # Get dimensions - handle Dict observation spaces
        self.state_dim, self.action_dim = self._get_dimensions()
        
    def _get_dimensions(self) -> Tuple[int, int]:
        """
        Get state and action dimensions from environment.
        
        Returns:
            (state_dim, action_dim)
        """
        # Action dimension
        if hasattr(self.env.action_space, 'shape'):
            action_dim = self.env.action_space.shape[0]
        else:
            raise ValueError(f"Cannot determine action dimension for {self.env_name}")
        
        # State dimension - handle different observation space types
        obs_space = self.env.observation_space
        
        if isinstance(obs_space, gym.spaces.Dict):
            # For Dict spaces (PointMaze), we need the 'observation' part
            # PointMaze: {'observation': Box(4,), 'desired_goal': Box(2,), 'achieved_goal': Box(2,)}
            # State for dynamics: just the observation (position + velocity)
            if 'observation' in obs_space.spaces:
                state_dim = obs_space.spaces['observation'].shape[0]
            else:
                raise ValueError(f"Dict observation space doesn't have 'observation' key: {obs_space.spaces.keys()}")
        
        elif isinstance(obs_space, gym.spaces.Box):
            # Simple Box space (HalfCheetah, Hopper, etc.)
            state_dim = obs_space.shape[0]
        
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")
        
        return state_dim, action_dim
    
    def get_dynamics(self, linearization_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract (A, B) matrices.
        
        Args:
            linearization_point: State to linearize around (if nonlinear)
        
        Returns:
            A: State transition matrix (state_dim, state_dim)
            B: Control matrix (state_dim, action_dim)
        """
        raise NotImplementedError
    
    def close(self):
        """Clean up environment."""
        self.env.close()


class AnalyticalDynamicsExtractor(DynamicsExtractor):
    """
    For environments with known analytical dynamics.
    E.g., PointMaze (double integrator), simple pendulum, etc.
    """
    
    def get_dynamics(self, linearization_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get analytical dynamics matrices."""
        
        # Check if we have analytical dynamics for this environment
        if 'maze' in self.env_name.lower() or 'pointmaze' in self.env_name.lower():
            return self._double_integrator_dynamics()
        else:
            raise ValueError(f"No analytical dynamics available for {self.env_name}")
    
    def _double_integrator_dynamics(self, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Double integrator dynamics (for PointMaze).
        
        State: [x, y, vx, vy]  (4D)
        Action: [ax, ay]       (2D)
        
        Dynamics:
            x_{t+1} = x_t + vx_t * dt
            y_{t+1} = y_t + vy_t * dt
            vx_{t+1} = vx_t + ax_t * dt
            vy_{t+1} = vy_t + ay_t * dt
        
        Returns:
            A: (4, 4) state transition matrix
            B: (4, 2) control matrix
        """
        # Verify dimensions
        if self.state_dim != 4:
            print(f"Warning: Expected state_dim=4 for PointMaze, got {self.state_dim}")
            print(f"Using double integrator with state_dim={self.state_dim}")
        
        if self.action_dim != 2:
            print(f"Warning: Expected action_dim=2 for PointMaze, got {self.action_dim}")
        
        # Double integrator matrices
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        B = np.array([
            [0.5 * dt**2, 0],
            [0, 0.5 * dt**2],
            [dt, 0],
            [0, dt]
        ])
        
        return A, B


class NumericalDynamicsExtractor(DynamicsExtractor):
    """
    For environments where we linearize numerically.
    Uses finite differences to compute Jacobians.
    """
    
    def get_dynamics(self, linearization_point: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numerically compute (A, B) via finite differences.
        
        Args:
            linearization_point: State to linearize around (if None, use reset state)
        
        Returns:
            A, B: Linearized dynamics matrices
        """
        if linearization_point is None:
            # Use default: reset state with zero velocity
            obs, _ = self.env.reset()
            linearization_point = self._extract_state(obs)
            
            # For PointMaze, ensure zero velocity for linearization
            if len(linearization_point) == 4:
                linearization_point[2:] = 0.0  # Zero velocity
        
        print(f"Linearizing around state: {linearization_point}")
        
        # Linearize around this point
        A = self._compute_A_matrix(linearization_point)
        B = self._compute_B_matrix(linearization_point)
        
        print(f"Computed A:\n{A}")
        print(f"Computed B:\n{B}")
        
        return A, B
    
    def _extract_state(self, obs):
        """Extract state vector from observation."""
        if isinstance(obs, dict):
            # For Dict observations (PointMaze), use 'observation' key
            if 'observation' in obs:
                state = obs['observation']
                # For PointMaze, state is [x, y, vx, vy]
                if len(state) >= self.state_dim:
                    return state[:self.state_dim].copy()
                else:
                    return state.copy()
            else:
                raise ValueError("Cannot extract state from dict observation")
        else:
            # Simple array observation
            return obs[:self.state_dim].copy()
    
    def _set_state(self, state: np.ndarray):
        """
        Set environment state (for finite differences).
        
        For PointMaze, we need to manipulate the internal state.
        """
        # PointMaze doesn't have set_state, so we'll use reset + manual setting
        # This is a workaround for environments without set_state
        
        if hasattr(self.env.unwrapped, 'set_state'):
            # MuJoCo-style environments
            n_qpos = self.state_dim // 2
            qpos = state[:n_qpos]
            qvel = state[n_qpos:]
            self.env.unwrapped.set_state(qpos, qvel)
            
        elif hasattr(self.env.unwrapped, 'state'):
            # PointMaze-style (Gymnasium robotics)
            # Directly set the agent's state
            if hasattr(self.env.unwrapped, 'model'):
                # MuJoCo environment wrapped by Gymnasium
                self.env.unwrapped.data.qpos[:2] = state[:2]  # Position
                self.env.unwrapped.data.qvel[:2] = state[2:4] if len(state) >= 4 else [0, 0]  # Velocity
                self.env.unwrapped.model.forward()
            else:
                raise NotImplementedError(f"Cannot set state for {self.env_name}")
        else:
            raise NotImplementedError(f"Cannot set state for {self.env_name}")
    
    def _step_dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Step the environment dynamics forward one step.
        
        Args:
            state: Current state [x, y, vx, vy]
            action: Action to take [ax, ay]
        
        Returns:
            next_state: Next state after dynamics
        """
        try:
            # Set the state
            self._set_state(state)
            
            # Step with action
            obs, _, _, _, _ = self.env.step(action)
            
            # Extract next state
            next_state = self._extract_state(obs)
            
            return next_state
            
        except NotImplementedError:
            # Fallback: Use analytical approximation if set_state not available
            print(f"Warning: set_state not available for {self.env_name}, using simplified dynamics")
            # This won't work perfectly, but better than crashing
            # Just step from current position
            obs, _ = self.env.reset()
            obs, _, _, _, _ = self.env.step(action)
            return self._extract_state(obs)
    
    def _compute_A_matrix(self, x0: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Compute A matrix: ∂f/∂x at x0 via finite differences.
        
        Increased eps from 1e-6 to 1e-4 for better numerical stability.
        """
        A = np.zeros((self.state_dim, self.state_dim))
        u0 = np.zeros(self.action_dim)
        
        # Get nominal next state
        x_nominal = self._step_dynamics(x0, u0)
        
        # Finite differences for each state dimension
        for i in range(self.state_dim):
            x_perturb = x0.copy()
            x_perturb[i] += eps
            
            x_next_perturb = self._step_dynamics(x_perturb, u0)
            
            # Finite difference derivative
            A[:, i] = (x_next_perturb - x_nominal) / eps
        
        return A
    
    def _compute_B_matrix(self, x0: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Compute B matrix: ∂f/∂u at x0 via finite differences.
        
        Increased eps from 1e-6 to 1e-4 for better numerical stability.
        """
        B = np.zeros((self.state_dim, self.action_dim))
        u0 = np.zeros(self.action_dim)
        
        # Get nominal next state
        x_nominal = self._step_dynamics(x0, u0)
        
        # Finite differences for each action dimension
        for i in range(self.action_dim):
            u_perturb = u0.copy()
            u_perturb[i] += eps
            
            x_next_perturb = self._step_dynamics(x0, u_perturb)
            
            # Finite difference derivative
            B[:, i] = (x_next_perturb - x_nominal) / eps
        
        return B

class TrajectoryDynamicsExtractor(DynamicsExtractor):
    """
    Extract dynamics by fitting to trajectory data.
    Uses least-squares: min ||X_next - [A B] * [X; U]||^2
    """
    
    def get_dynamics(self, 
                     num_trajectories: int = 1000,
                     trajectory_length: int = 80,
                     use_dataset: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit (A, B) matrices from trajectory data.
        
        Args:
            num_trajectories: Number of trajectories to collect
            trajectory_length: Length of each trajectory
            use_dataset: Optional path to existing dataset (e.g., D4RL dataset)
        
        Returns:
            A, B: Fitted dynamics matrices
        """
        if use_dataset is not None:
            # Load from existing dataset (D4RL, Minari, etc.)
            states, actions, next_states = self._load_dataset(use_dataset)
        else:
            # Collect new trajectories
            states, actions, next_states = self._collect_trajectories(
                num_trajectories, trajectory_length
            )
        
        # Fit dynamics using least squares
        A, B = self._fit_linear_dynamics(states, actions, next_states)
        
        return A, B
    
    def _collect_trajectories(self, 
                            num_traj: int, 
                            traj_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect trajectory data from environment.
        
        Returns:
            states: (N, state_dim)
            actions: (N, action_dim)
            next_states: (N, state_dim)
        """
        all_states = []
        all_actions = []
        all_next_states = []
        
        print(f"Collecting {num_traj} trajectories of length {traj_len}...")
        
        for traj_idx in range(num_traj):
            obs, _ = self.env.reset()
            state = self._extract_state(obs)
            
            for step in range(traj_len):
                # Random action (or could use a policy)
                action = self.env.action_space.sample()
                
                # Step environment
                next_obs, _, terminated, truncated, _ = self.env.step(action)
                next_state = self._extract_state(next_obs)
                
                # Store transition
                all_states.append(state)
                all_actions.append(action)
                all_next_states.append(next_state)
                
                # Update state
                state = next_state
                
                if terminated or truncated:
                    break
            
            if (traj_idx + 1) % 10 == 0:
                print(f"  Collected {traj_idx + 1}/{num_traj} trajectories")
        
        states = np.array(all_states)
        actions = np.array(all_actions)
        next_states = np.array(all_next_states)
        
        print(f"Collected {len(states)} transitions")
        
        return states, actions, next_states
    
    def _load_dataset(self, dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load trajectory data from existing dataset.
        
        For D4RL/Minari datasets.
        """
        try:
            import minari
            dataset = minari.load_dataset(dataset_path)
            
            all_states = []
            all_actions = []
            all_next_states = []
            
            print(f"Loading dataset: {dataset_path}")
            
            for episode_idx, episode in enumerate(dataset):
                observations = episode.observations
                actions = episode.actions
                
                # Handle dict observations (PointMaze)
                if isinstance(observations, dict):
                    if 'observation' in observations:
                        # Extract state trajectory (T+1, state_dim)
                        obs_array = observations['observation']
                    else:
                        print(f"Warning: Skipping episode {episode_idx} - unknown dict structure")
                        continue
                else:
                    obs_array = observations
                
                # Create transitions: (s_t, a_t, s_{t+1})
                for t in range(len(actions)):
                    state = obs_array[t]
                    action = actions[t]
                    next_state = obs_array[t + 1]
                    
                    all_states.append(state)
                    all_actions.append(action)
                    all_next_states.append(next_state)
            
            states = np.array(all_states)
            actions = np.array(all_actions)
            next_states = np.array(all_next_states)
            
            print(f"Loaded {len(states)} transitions from dataset")
            
            return states, actions, next_states
            
        except Exception as e:
            print(f"Could not load dataset: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to trajectory collection")
            return self._collect_trajectories(num_traj=100, traj_len=50)
        
    def _fit_linear_dynamics(self, 
                            states: np.ndarray,
                            actions: np.ndarray, 
                            next_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit linear dynamics using least squares.
        
        Model: x_{t+1} = A*x_t + B*u_t
        
        Solve: min ||X_next - [A B] * [X; U]^T||^2
        
        Args:
            states: (N, state_dim)
            actions: (N, action_dim)
            next_states: (N, state_dim)
        
        Returns:
            A: (state_dim, state_dim)
            B: (state_dim, action_dim)
        """
        N = states.shape[0]
        
        # Construct regressor matrix: [X, U]
        # Shape: (N, state_dim + action_dim)
        X = np.hstack([states, actions])
        
        # Target: X_next
        # Shape: (N, state_dim)
        Y = next_states
        
        # Solve least squares: [A, B] = argmin ||Y - X @ [A, B]^T||^2
        # This gives us [A, B]^T
        # Shape: (state_dim + action_dim, state_dim)
        AB_T = np.linalg.lstsq(X, Y, rcond=None)[0]
        
        # Split into A and B
        A = AB_T[:self.state_dim, :].T  # (state_dim, state_dim)
        B = AB_T[self.state_dim:, :].T   # (state_dim, action_dim)
        
        # Compute fit quality (R^2)
        Y_pred = X @ AB_T
        residuals = Y - Y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((Y - Y.mean(axis=0))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nLeast squares fit:")
        print(f"  R^2 = {r_squared:.6f}")
        print(f"  Mean prediction error = {np.mean(np.linalg.norm(residuals, axis=1)):.6f}")
        print(f"  Max prediction error = {np.max(np.linalg.norm(residuals, axis=1)):.6f}")
        
        return A, B
    
    def _extract_state(self, obs):
        """Extract state vector from observation."""
        if isinstance(obs, dict):
            if 'observation' in obs:
                return obs['observation'].copy()
            else:
                raise ValueError("Cannot extract state from dict observation")
        else:
            return obs.copy()



def get_dynamics_extractor(env_name: str, method: str = 'auto') -> DynamicsExtractor:
    """
    Factory function to get appropriate dynamics extractor.
    
    Args:
        env_name: Gymnasium environment name
        method: 'analytical', 'numerical', 'trajectory', or 'auto'
    
    Returns:
        DynamicsExtractor instance
    """
    if method == 'auto':
        # Automatically choose based on environment
        if any(name in env_name.lower() for name in ['maze', 'pointmaze']):
            method = 'analytical'  # Analytical is fastest if available
        else:
            method = 'trajectory'  # Default to trajectory fitting
    
    if method == 'analytical':
        return AnalyticalDynamicsExtractor(env_name)
    elif method == 'numerical':
        return NumericalDynamicsExtractor(env_name)
    elif method == 'trajectory':
        return TrajectoryDynamicsExtractor(env_name)
    else:
        raise ValueError(f"Unknown method: {method}")