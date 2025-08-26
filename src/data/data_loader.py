import os
import random
import numpy as np
import torch
from typing import List, Tuple, Optional


class TrajectoryDataLoader:
    """Unified data loader for trajectory data across all experiments."""
    
    def __init__(self, data_dir: str, state_dim: int = 4, control_dim: int = 2):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing trajectory .npz files
            state_dim: Dimension of state vectors
            control_dim: Dimension of control vectors
        """
        self.data_dir = data_dir
        self.state_dim = state_dim
        self.control_dim = control_dim
        
        # Verify directory exists
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory {data_dir} does not exist")
    
    def load_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load a random batch of trajectories for training.
        
        Args:
            batch_size: Number of trajectories to load
            
        Returns:
            trajectories: [batch_size, seq_len, state_dim + control_dim]
            conditions: [batch_size, 2 * state_dim] (initial + target states)
        """
        file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        
        if len(file_list) == 0:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        # Randomly select files for batch
        selected_files = random.sample(file_list, min(batch_size, len(file_list)))
        return self._load_files(selected_files)
    
    def load_specific(self, 
                     file_indices: Optional[List[int]] = None, 
                     file_names: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Load specific trajectories for evaluation.
        
        Args:
            file_indices: List of indices to select from sorted file list
            file_names: List of specific file names to load
            
        Returns:
            trajectories: [num_files, seq_len, state_dim + control_dim]
            conditions: [num_files, 2 * state_dim]
            loaded_file_names: List of successfully loaded file names
        """
        file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        
        if len(file_list) == 0:
            raise ValueError(f"No .npz files found in {self.data_dir}")
        
        # Determine which files to load
        if file_indices is not None:
            selected_files = [file_list[i] for i in file_indices if i < len(file_list)]
        elif file_names is not None:
            selected_files = [f for f in file_names if f in file_list]
        else:
            selected_files = file_list
        
        if not selected_files:
            raise ValueError("No valid files selected")
        
        print(f"Loading {len(selected_files)} specific trajectories")
        
        trajectories, conditions = self._load_files(selected_files)
        return trajectories, conditions, selected_files
    
    def _load_files(self, file_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal method to load trajectory files.
        
        Args:
            file_names: List of file names to load
            
        Returns:
            trajectories: Tensor of loaded trajectories
            conditions: Tensor of loaded conditions
        """
        trajectories = []
        conditions = []
        
        for file_name in file_names:
            try:
                file_path = os.path.join(self.data_dir, file_name)
                data = np.load(file_path)
                
                # Extract states and controls
                states = data['states'][:-1]  # Remove last state to match control length
                controls = data['controls']
                initial_state = data['initial_state']
                target_state = data['target_state']
                
                # Validate dimensions
                if states.shape[1] != self.state_dim:
                    print(f"Warning: State dimension mismatch in {file_name}. "
                          f"Expected {self.state_dim}, got {states.shape[1]}")
                    continue
                    
                if controls.shape[1] != self.control_dim:
                    print(f"Warning: Control dimension mismatch in {file_name}. "
                          f"Expected {self.control_dim}, got {controls.shape[1]}")
                    continue
                
                # Combine states and controls into single trajectory
                trajectory = np.concatenate([states, controls], axis=1)
                
                # Combine initial and target states as conditioning
                condition = np.concatenate([initial_state, target_state])
                
                trajectories.append(trajectory)
                conditions.append(condition)
                
            except Exception as e:
                print(f"Error loading file {file_name}: {e}")
                continue
        
        if not trajectories:
            raise ValueError("No valid trajectories could be loaded")
        
        # Convert to tensors
        trajectories_tensor = torch.tensor(np.array(trajectories), dtype=torch.float32)
        conditions_tensor = torch.tensor(np.array(conditions), dtype=torch.float32)
        
        return trajectories_tensor, conditions_tensor
    
    def get_data_info(self) -> dict:
        """Get information about the dataset."""
        file_list = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        
        if not file_list:
            return {"num_files": 0, "seq_len": None, "feature_dim": None}
        
        # Load one file to get dimensions
        sample_data = np.load(os.path.join(self.data_dir, file_list[0]))
        seq_len = len(sample_data['states']) - 1  # Subtract 1 for matching with controls
        feature_dim = self.state_dim + self.control_dim
        
        return {
            "num_files": len(file_list),
            "seq_len": seq_len,
            "feature_dim": feature_dim,
            "state_dim": self.state_dim,
            "control_dim": self.control_dim
        }