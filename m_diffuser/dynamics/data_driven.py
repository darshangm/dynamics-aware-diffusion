"""
Data-driven system identification from trajectories.
Uses least-squares to fit linear dynamics from data.
"""

import numpy as np
from typing import Tuple, List, Dict
import minari


def extract_transitions(dataset_name: str, max_trajectories: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (state, action, next_state) transitions from dataset.
    
    Args:
        dataset_name: Minari dataset name
        max_trajectories: Maximum number of trajectories to use
    
    Returns:
        states: (N, state_dim)
        actions: (N, action_dim)
        next_states: (N, state_dim)
    """
    print(f"Loading dataset for system ID: {dataset_name}")
    dataset = minari.load_dataset(dataset_name)
    
    states_list = []
    actions_list = []
    next_states_list = []
    
    num_episodes = min(len(dataset), max_trajectories)
    print(f"Using {num_episodes} trajectories for system identification")
    
    for episode_idx, episode in enumerate(dataset):
        if episode_idx >= max_trajectories:
            break
        
        # Extract observations
        obs = episode.observations
        
        # Handle dict observations (PointMaze)
        if isinstance(obs, dict):
            if 'observation' in obs:
                obs_array = obs['observation']  # (T+1, state_dim)
            else:
                print(f"Warning: Cannot extract observations from episode {episode_idx}")
                continue
        else:
            obs_array = obs
        
        # Extract actions
        actions = np.array(episode.actions)  # (T, action_dim)
        
        # Create transitions: (s_t, a_t, s_{t+1})
        for t in range(len(actions)):
            state_t = obs_array[t]
            action_t = actions[t]
            state_t1 = obs_array[t + 1]
            
            states_list.append(state_t)
            actions_list.append(action_t)
            next_states_list.append(state_t1)
    
    states = np.array(states_list)
    actions = np.array(actions_list)
    next_states = np.array(next_states_list)
    
    print(f"Extracted {len(states)} transitions")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    
    return states, actions, next_states


def fit_linear_dynamics(states: np.ndarray, 
                       actions: np.ndarray, 
                       next_states: np.ndarray,
                       state_dim: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit linear dynamics x_{t+1} = A·x_t + B·u_t using least squares.
    
    Args:
        states: (N, state_dim) current states
        actions: (N, action_dim) actions
        next_states: (N, state_dim) next states
        state_dim: If provided, only use first state_dim dimensions
    
    Returns:
        A: (state_dim, state_dim) state transition matrix
        B: (state_dim, action_dim) control matrix
    """
    # Use only the relevant state dimensions (e.g., exclude goal for PointMaze)
    if state_dim is not None and states.shape[1] > state_dim:
        print(f"Using first {state_dim} dimensions of {states.shape[1]}-dim observations")
        states = states[:, :state_dim]
        next_states = next_states[:, :state_dim]
    
    N = states.shape[0]
    n = states.shape[1]  # state_dim
    m = actions.shape[1]  # action_dim
    
    print(f"\nFitting linear dynamics:")
    print(f"  N = {N} data points")
    print(f"  State dim = {n}")
    print(f"  Action dim = {m}")
    
    # Build regression matrix: [X, U]
    # Shape: (N, n+m)
    Phi = np.hstack([states, actions])
    
    # Target: X_next
    # Shape: (N, n)
    Y = next_states
    
    # Least squares: Θ = (Φᵀ Φ)^{-1} Φᵀ Y
    # Θ shape: (n+m, n)
    Theta = np.linalg.lstsq(Phi, Y, rcond=None)[0]
    
    # Extract A and B
    A = Theta[:n, :].T   # (n, n)
    B = Theta[n:, :].T   # (n, m)
    
    # Compute fit quality (R²)
    Y_pred = Phi @ Theta
    residuals = Y - Y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y - Y.mean(axis=0))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"\nLeast-squares fit quality:")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Mean prediction error = {np.mean(np.linalg.norm(residuals, axis=1)):.6f}")
    
    return A, B


def identify_dynamics_from_data(dataset_name: str, 
                                state_dim: int = None,
                                max_trajectories: int = 1000) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Identify linear dynamics (A, B) from dataset using least squares.
    
    Args:
        dataset_name: Minari dataset name
        state_dim: Dimension of physical state (exclude goal, etc.)
        max_trajectories: Maximum trajectories to use
    
    Returns:
        A: State transition matrix
        B: Control matrix
        state_dim: State dimension
        action_dim: Action dimension
    """
    # Extract transitions
    states, actions, next_states = extract_transitions(dataset_name, max_trajectories)
    
    # Infer dimensions if not provided
    if state_dim is None:
        state_dim = states.shape[1]
    
    action_dim = actions.shape[1]
    
    # Fit linear dynamics
    A, B = fit_linear_dynamics(states, actions, next_states, state_dim)
    
    return A, B, state_dim, action_dim