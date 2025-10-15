"""
Registry mapping environment names to dynamics extraction methods.
"""

from typing import Tuple
import numpy as np
from .extractor import get_dynamics_extractor
from .data_driven import identify_dynamics_from_data


# Environment name patterns → dynamics type
DYNAMICS_REGISTRY = {
    'pointmaze': 'data_driven',      # NEW: Use data-driven ID
    'maze': 'data_driven',            # NEW: Use data-driven ID
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


def get_dynamics_for_env(env_name: str, 
                        dataset_name: str = None,
                        linearization_point: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Get dynamics matrices (A, B) for an environment.
    
    Args:
        env_name: Gymnasium environment name
        dataset_name: Minari dataset name (for data-driven ID)
        linearization_point: Optional state to linearize around (for numerical)
    
    Returns:
        A: State transition matrix
        B: Control matrix  
        state_dim: State dimension
        action_dim: Action dimension
    """
    # Determine extraction method
    method = 'numerical'  # default
    for pattern, dynamics_type in DYNAMICS_REGISTRY.items():
        if pattern.lower() in env_name.lower():
            method = dynamics_type
            break
    
    # Data-driven system ID (preferred!)
    if method == 'data_driven' and dataset_name is not None:
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
    
    # Fallback to numerical linearization
    print(f"Using numerical linearization (method={method})")
    extractor = get_dynamics_extractor(env_name, method=method)
    A, B = extractor.get_dynamics(linearization_point)
    
    state_dim = extractor.state_dim
    action_dim = extractor.action_dim
    
    extractor.close()
    
    return A, B, state_dim, action_dim