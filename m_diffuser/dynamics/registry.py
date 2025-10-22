"""
Registry mapping environment names to dynamics extraction methods.
"""

from typing import Tuple
import numpy as np
from .extractor import get_dynamics_extractor
from .data_driven import identify_dynamics_from_data


# Environment name patterns → dynamics type
DYNAMICS_REGISTRY = {
    'pointmaze': 'data_driven',      # Use data-driven from Minari dataset
    'maze': 'data_driven',            
    'halfcheetah': 'data_driven',
    'hopper': 'data_driven',
    'walker': 'data_driven',
}

# State dimensions (physical state, excluding goals etc.)
STATE_DIM_REGISTRY = {
    'pointmaze': 4,  # [x, y, vx, vy] (exclude goal)
    'maze': 4,
    'halfcheetah': 17,
    'hopper': 11,
    'walker': 17,
}

# Map environment names to Minari dataset names
DATASET_REGISTRY = {
    'pointmaze_umaze': 'D4RL/pointmaze/umaze-v2',
    'pointmaze_medium': 'D4RL/pointmaze/medium-v2',
    'pointmaze_large': 'D4RL/pointmaze/large-v2',
}


def get_dynamics_for_env(env_name: str, 
                        dataset_name: str = None,
                        method: str = None,
                        linearization_point: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Get dynamics matrices (A, B) for an environment.
    
    Args:
        env_name: Gymnasium environment name
        dataset_name: Minari dataset name (for data-driven ID)
        method: Override automatic method selection ('data_driven', 'analytical', 'trajectory', 'numerical')
        linearization_point: Optional state to linearize around (for numerical)
    
    Returns:
        A: State transition matrix
        B: Control matrix  
        state_dim: State dimension
        action_dim: Action dimension
    """
    # Determine extraction method
    if method is None:
        method = 'numerical'  # default fallback
        for pattern, dynamics_type in DYNAMICS_REGISTRY.items():
            if pattern.lower() in env_name.lower():
                method = dynamics_type
                break
    
    print(f"Extracting dynamics for {env_name} using method: {method}")
    
    # Data-driven system ID from Minari dataset (preferred!)
    if method == 'data_driven':
        if dataset_name is None:
            # Try to infer dataset name from environment
            env_key = env_name.lower().replace('-', '_').replace('_v3', '')
            if env_key in DATASET_REGISTRY:
                dataset_name = DATASET_REGISTRY[env_key]
                print(f"Inferred dataset name: {dataset_name}")
            else:
                print(f"Warning: No dataset specified and couldn't infer from env name")
                print(f"Falling back to analytical/trajectory method")
                method = 'analytical' if 'maze' in env_name.lower() else 'trajectory'
        
        if dataset_name is not None:
            print(f"Using data-driven system identification from {dataset_name}")
            
            # Get state dimension
            state_dim = None
            for pattern, dim in STATE_DIM_REGISTRY.items():
                if pattern.lower() in env_name.lower():
                    state_dim = dim
                    break
            
            A, B, state_dim, action_dim = identify_dynamics_from_data(
                dataset_name, 
                state_dim=state_dim,
                max_trajectories=1000
            )
            
            return A, B, state_dim, action_dim
    
    # Fallback to other methods
    print(f"Using {method} method")
    extractor = get_dynamics_extractor(env_name, method=method)
    
    if method == 'trajectory' and dataset_name is not None:
        # Use dataset if available
        A, B = extractor.get_dynamics(use_dataset=dataset_name)
    else:
        A, B = extractor.get_dynamics(linearization_point)
    
    state_dim = extractor.state_dim
    action_dim = extractor.action_dim
    
    extractor.close()
    
    return A, B, state_dim, action_dim