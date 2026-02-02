"""
Sequence dataset for trajectory data.
Adapted from original Diffuser to work with Minari/Gymnasium.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import minari

from .normalization import DatasetNormalizer


class SequenceDataset(Dataset):
    """
    Dataset that returns fixed-length trajectory sequences.
    
    Original Diffuser used D4RL, we use Minari for modern Gymnasium environments.
    """
    
    def __init__(self,
                dataset_name: str,
                horizon: int = 64,
                normalizer: str = 'LimitsNormalizer',
                max_path_length: int = 1000,
                use_padding: bool = True,
                discount: float = 0.99,
                returns_scale: float = 1000.0):
        """
        Initialize sequence dataset.
        
        Args:
            dataset_name: Minari dataset name (e.g., 'halfcheetah-medium-v0')
            horizon: Length of trajectory sequences
            normalizer: Type of normalizer ('LimitsNormalizer' or 'GaussianNormalizer')
            max_path_length: Maximum episode length
            use_padding: Whether to pad sequences at episode boundaries
            discount: Discount factor for returns calculation
            returns_scale: Scale factor for returns
        """
        self.dataset_name = dataset_name
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.discount = discount
        self.returns_scale = returns_scale
        
        # Load dataset from Minari
        print(f"Loading dataset: {dataset_name}")
        self.minari_dataset = minari.load_dataset(dataset_name)
        
        # Extract trajectories (now only extracts physical states, no goals)
        self.trajectories = self._load_trajectories()
        
        # Flatten trajectories into sequences
        self.observations, self.actions, self.rewards = self._flatten_trajectories()
        
        # Create normalizer
        self.observation_dim = self.observations.shape[-1]  # Physical state only
        self.action_dim = self.actions.shape[-1]
        self.transition_dim = self.observation_dim + self.action_dim
        
        print(f"Dataset dimensions:")
        print(f"  Observation dim (physical state): {self.observation_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Transition dim: {self.transition_dim}")
        
        # Single normalizer for states and actions
        self.normalizer = DatasetNormalizer(
            observations=self.observations,
            actions=self.actions,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim
        )
        
        # Normalize data
        self.normed_observations = self.normalizer.normalize_observations(self.observations)
        self.normed_actions = self.normalizer.normalize_actions(self.actions)
        
        # Create indices for sampling
        self.indices = self._make_indices()
        
        print(f"Dataset loaded: {len(self.trajectories)} trajectories, "
            f"{len(self.indices)} sequences")

    def _load_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Load trajectories from Minari dataset."""
        trajectories = []
        
        for episode_idx, episode in enumerate(self.minari_dataset):
            try:
                # Handle different observation structures
                obs = episode.observations
                
                # Debug: print structure of first episode
                if episode_idx == 0:
                    print(f"Debug - Episode observations type: {type(obs)}")
                    if isinstance(obs, dict):
                        print(f"Debug - Dictionary keys: {obs.keys()}")
                
                # Handle dict observations (PointMaze) - EXTRACT ONLY STATE
                if isinstance(obs, dict):
                    if 'observation' in obs:
                        # Use ONLY the physical state, ignore goals!
                        obs = obs['observation']  # (T+1, 4) for PointMaze
                        print(f"Debug - Using only 'observation' key, shape: {obs.shape}") if episode_idx == 0 else None
                    else:
                        print(f"Warning: Skipping episode {episode_idx} - no 'observation' key")
                        continue
                
                # Convert to numpy array if not already
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                
                # Check if valid
                if obs.ndim == 0 or obs.size == 0:
                    print(f"Warning: Skipping episode {episode_idx} - empty observations")
                    continue
                
                # Ensure 2D array (timesteps, features)
                if obs.ndim == 1:
                    obs = obs.reshape(-1, 1)
                
                actions = np.array(episode.actions)
                if actions.ndim == 1:
                    actions = actions.reshape(-1, 1)
                    
                rewards = np.array(episode.rewards)
                
                # Make sure we have valid data
                if len(obs) <= 1 or len(actions) == 0:
                    print(f"Warning: Skipping episode {episode_idx} - insufficient data")
                    continue
                
                # Minari stores next observations separately
                # We need obs[:-1] to match with actions
                trajectory = {
                    'observations': obs[:-1],  # ONLY physical state [x, y, vx, vy]
                    'actions': actions,
                    'rewards': rewards,
                    'terminals': np.zeros(len(actions), dtype=bool)
                }
                
                # Mark terminal state
                trajectory['terminals'][-1] = True
                
                trajectories.append(trajectory)
                
            except Exception as e:
                print(f"Error processing episode {episode_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
        if len(trajectories) == 0:
            raise ValueError("No valid trajectories loaded from dataset!")
        
        print(f"Successfully loaded {len(trajectories)} trajectories")
        return trajectories

    def _flatten_trajectories(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flatten all trajectories into single arrays."""
        observations = []
        actions = []
        rewards = []
        
        for traj in self.trajectories:
            observations.append(traj['observations'])
            actions.append(traj['actions'])
            rewards.append(traj['rewards'])
        
        return (
            np.concatenate(observations, axis=0),
            np.concatenate(actions, axis=0),
            np.concatenate(rewards, axis=0)
        )

    
    def _make_indices(self) -> List[Tuple[int, int, int]]:
        """
        Create valid indices for sampling sequences.
        
        Returns:
            List of (trajectory_idx, start_idx, end_idx) tuples
        """
        indices = []
        
        for traj_idx, traj in enumerate(self.trajectories):
            path_length = len(traj['observations'])
            
            if path_length < self.horizon:
                # If trajectory is shorter than horizon, pad it
                if self.use_padding:
                    indices.append((traj_idx, 0, path_length))
            else:
                # Sample all possible windows
                max_start = path_length - self.horizon + 1
                for start in range(max_start):
                    end = start + self.horizon
                    indices.append((traj_idx, start, end))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a trajectory sequence.
        
        Returns:
            Dictionary containing:
                - observations: (horizon, obs_dim)
                - actions: (horizon, action_dim)
                - rewards: (horizon,)
        """
        traj_idx, start_idx, end_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]
        
        # Extract sequence
        obs_seq = traj['observations'][start_idx:end_idx]
        action_seq = traj['actions'][start_idx:end_idx]
        reward_seq = traj['rewards'][start_idx:end_idx]
        
        # Pad if necessary
        if len(obs_seq) < self.horizon and self.use_padding:
            pad_length = self.horizon - len(obs_seq)
            obs_seq = np.concatenate([
                obs_seq,
                np.zeros((pad_length, self.observation_dim))
            ], axis=0)
            action_seq = np.concatenate([
                action_seq,
                np.zeros((pad_length, self.action_dim))
            ], axis=0)
            reward_seq = np.concatenate([
                reward_seq,
                np.zeros(pad_length)
            ], axis=0)
        
        # Normalize
        normed_obs = self.normalizer.normalize_observations(obs_seq)
        normed_actions = self.normalizer.normalize_actions(action_seq)
        
        # Combine observations and actions into "conditions"
        # In Diffuser, both obs and actions are diffused together
        conditions = np.concatenate([normed_obs, normed_actions], axis=-1)
        
        return {
            'conditions': torch.FloatTensor(conditions),
            'observations': torch.FloatTensor(normed_obs),
            'actions': torch.FloatTensor(normed_actions),
            'rewards': torch.FloatTensor(reward_seq)
        }
    
    def get_conditions(self, observations: np.ndarray) -> np.ndarray:
        """
        Convert observations to conditions for planning.
        Used during inference.
        """
        normed_obs = self.normalizer.normalize_observations(observations)
        return normed_obs


def create_dataloader(dataset: SequenceDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0) -> torch.utils.data.DataLoader:
    """Create a dataloader for the sequence dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )