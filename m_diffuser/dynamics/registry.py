"""
Registry mapping environment names to dynamics extraction methods.
"""

from typing import Tuple
import numpy as np
from .extractor import get_dynamics_extractor


# Environment name patterns → dynamics type
DYNAMICS_REGISTRY = {
    'pointmaze': 'analytical',
    'maze': 'analytical',
    'halfcheetah': 'numerical',
    'hopper': 'numerical',
    'walker': 'numerical',
    'ant': 'numerical',
    'adroithand': 'numerical',
}


def get_dynamics_for_env(env_name: str, linearization_point: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Get dynamics matrices (A, B) for an environment.
    
    Args:
        env_name: Gymnasium environment name
        linearization_point: Optional state to linearize around
    
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
    
    # Extract dynamics
    extractor = get_dynamics_extractor(env_name, method=method)
    A, B = extractor.get_dynamics(linearization_point)
    
    state_dim = extractor.state_dim
    action_dim = extractor.action_dim
    
    extractor.close()
    
    return A, B, state_dim, action_dim