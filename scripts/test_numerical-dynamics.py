"""Test numerical linearization for PointMaze."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
import gymnasium_robotics
from m_diffuser.dynamics import get_dynamics_for_env
import numpy as np



print("=" * 60)
print("Testing Numerical Linearization for PointMaze")
print("=" * 60)

# Extract dynamics numerically
A, B, state_dim, action_dim = get_dynamics_for_env('PointMaze_UMaze-v3')

print(f"\nExtracted dynamics:")
print(f"State dim: {state_dim}, Action dim: {action_dim}")
print(f"\nA matrix ({A.shape}):")
print(A)
print(f"\nB matrix ({B.shape}):")
print(B)

# Test prediction accuracy


env = gym.make('PointMaze_UMaze-v3')
obs, _ = env.reset(seed=42)
state = obs['observation'][:4].copy()

print(f"\n" + "=" * 60)
print("Testing Prediction Accuracy")
print("=" * 60)

action = np.array([0.5, 0.3])
errors = []

for step in range(10):
    # Predict using linearized model
    state_pred = A @ state + B @ action
    
    # Actual dynamics
    obs, _, _, _, _ = env.step(action)
    state_actual = obs['observation'][:4]
    
    error = np.linalg.norm(state_pred - state_actual)
    errors.append(error)
    
    print(f"Step {step}: error = {error:.6f}")
    
    state = state_actual

print(f"\nMean error: {np.mean(errors):.6f}")
print(f"Max error: {np.max(errors):.6f}")

env.close()

if np.mean(errors) < 0.01:
    print("\n✓ Numerical linearization is accurate!")
else:
    print(f"\n⚠ Linearization error is high ({np.mean(errors):.6f})")
    print("This might affect dynamics-aware performance")