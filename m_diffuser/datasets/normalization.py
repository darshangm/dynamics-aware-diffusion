"""
Data normalization for trajectory datasets.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DatasetNormalizer:
    """
    Normalizes observations and actions in trajectory data.
    Uses mean/std normalization by default.
    """
    
    def __init__(self, 
                 observations: np.ndarray,
                 actions: np.ndarray,
                 observation_dim: int,
                 action_dim: int):
        """
        Initialize normalizer with dataset statistics.
        
        Args:
            observations: Array of observations (N, obs_dim)
            actions: Array of actions (N, action_dim)
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Compute normalization statistics
        self.obs_mean = observations.mean(axis=0)
        self.obs_std = observations.std(axis=0)
        
        self.action_mean = actions.mean(axis=0)
        self.action_std = actions.std(axis=0)
        
        # Prevent division by zero
        self.obs_std = np.maximum(self.obs_std, 1e-6)
        self.action_std = np.maximum(self.action_std, 1e-6)
        
    def normalize_observations(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations."""
        return (obs - self.obs_mean) / self.obs_std
    
    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions."""
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_observations(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize observations."""
        return obs * self.obs_std + self.obs_mean
    
    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize actions."""
        return actions * self.action_std + self.action_mean
    
    def normalize_trajectory(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize a full trajectory dictionary.
        
        Args:
            trajectory: Dict with 'observations' and 'actions' keys
            
        Returns:
            Normalized trajectory dictionary
        """
        normalized = {}
        if 'observations' in trajectory:
            normalized['observations'] = self.normalize_observations(trajectory['observations'])
        if 'actions' in trajectory:
            normalized['actions'] = self.normalize_actions(trajectory['actions'])
        if 'rewards' in trajectory:
            normalized['rewards'] = trajectory['rewards']  # Don't normalize rewards
        return normalized
    
    def unnormalize_trajectory(self, trajectory: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Unnormalize a full trajectory dictionary.
        
        Args:
            trajectory: Normalized dict with 'observations' and 'actions' keys
            
        Returns:
            Unnormalized trajectory dictionary
        """
        unnormalized = {}
        if 'observations' in trajectory:
            unnormalized['observations'] = self.unnormalize_observations(trajectory['observations'])
        if 'actions' in trajectory:
            unnormalized['actions'] = self.unnormalize_actions(trajectory['actions'])
        if 'rewards' in trajectory:
            unnormalized['rewards'] = trajectory['rewards']
        return unnormalized
    
    def get_statistics(self) -> Dict[str, np.ndarray]:
        """Return normalization statistics."""
        return {
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'action_mean': self.action_mean,
            'action_std': self.action_std
        }


class MinMaxNormalizer:
    """
    Normalizes data to [-1, 1] range using min-max scaling.
    Alternative to mean/std normalization.
    """
    
    def __init__(self,
                 observations: np.ndarray,
                 actions: np.ndarray,
                 observation_dim: int,
                 action_dim: int):
        """
        Initialize min-max normalizer.
        
        Args:
            observations: Array of observations (N, obs_dim)
            actions: Array of actions (N, action_dim)
            observation_dim: Dimension of observation space
            action_dim: Dimension of action space
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Compute min-max statistics
        self.obs_min = observations.min(axis=0)
        self.obs_max = observations.max(axis=0)
        
        self.action_min = actions.min(axis=0)
        self.action_max = actions.max(axis=0)
        
        # Prevent division by zero
        self.obs_range = np.maximum(self.obs_max - self.obs_min, 1e-6)
        self.action_range = np.maximum(self.action_max - self.action_min, 1e-6)
        
    def normalize_observations(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations to [-1, 1]."""
        normalized = (obs - self.obs_min) / self.obs_range
        return 2 * normalized - 1
    
    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Normalize actions to [-1, 1]."""
        normalized = (actions - self.action_min) / self.action_range
        return 2 * normalized - 1
    
    def unnormalize_observations(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize observations from [-1, 1]."""
        unnormalized = (obs + 1) / 2
        return unnormalized * self.obs_range + self.obs_min
    
    def unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Unnormalize actions from [-1, 1]."""
        unnormalized = (actions + 1) / 2
        return unnormalized * self.action_range + self.action_min