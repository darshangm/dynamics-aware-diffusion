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


def get_dynamics_extractor(env_name: str, method: str = 'auto') -> DynamicsExtractor:
    """
    Factory function to get appropriate dynamics extractor.
    
    Args:
        env_name: Gymnasium environment name
        method: 'analytical', 'numerical', or 'auto'
    
    Returns:
        DynamicsExtractor instance
    """
    if method == 'auto':
        # Automatically choose based on environment
        if any(name in env_name.lower() for name in ['maze', 'pointmaze']):
            method = 'analytical'
        else:
            method = 'numerical'
    
    if method == 'analytical':
        return AnalyticalDynamicsExtractor(env_name)
    elif method == 'numerical':
        return NumericalDynamicsExtractor(env_name)
    else:
        raise ValueError(f"Unknown method: {method}")