"""Test data-driven system identification."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from m_diffuser.dynamics.data_driven import identify_dynamics_from_data
import numpy as np

print("=" * 60)
print("Data-Driven System Identification Test")
print("=" * 60)

# Identify dynamics from PointMaze data
A, B, state_dim, action_dim = identify_dynamics_from_data(
    'D4RL/pointmaze/umaze-v2',
    state_dim=4,  # [x, y, vx, vy]
    max_trajectories=1000
)

print(f"\nIdentified dynamics:")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print(f"\nA matrix:")
print(A)
print(f"\nB matrix:")
print(B)

# These should look reasonable now!
print(f"\nA eigenvalues: {np.linalg.eigvals(A)}")
print(f"B norms: {np.linalg.norm(B, axis=0)}")