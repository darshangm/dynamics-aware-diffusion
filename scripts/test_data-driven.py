"""Test data-driven dynamics extraction for PointMaze."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from m_diffuser.dynamics import get_dynamics_for_env
import gymnasium as gym
import gymnasium_robotics

print("=" * 60)
print("Testing Data-Driven Dynamics Extraction")
print("=" * 60)

# Test with PointMaze
env_name = 'PointMaze_UMaze-v3'
dataset_name = 'D4RL/pointmaze/umaze-v2'

print(f"\nEnvironment: {env_name}")
print(f"Dataset: {dataset_name}")

# Extract dynamics from data
A, B, state_dim, action_dim = get_dynamics_for_env(
    env_name=env_name,
    dataset_name=dataset_name,
    method='data_driven'
)

print(f"\n" + "=" * 60)
print("Extracted Dynamics")
print("=" * 60)
print(f"\nState dim: {state_dim}, Action dim: {action_dim}")
print(f"\nA matrix ({A.shape}):")
print(A)
print(f"\nB matrix ({B.shape}):")
print(B)

# Compare with analytical
print(f"\n" + "=" * 60)
print("Comparison with Analytical Dynamics")
print("=" * 60)

dt = 0.1
mass = 1.0
A_analytical = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
B_analytical = np.array([
    [0.5 * dt**2 / mass, 0],
    [0, 0.5 * dt**2 / mass],
    [dt / mass, 0],
    [0, dt / mass]
])

print(f"\nAnalytical A:")
print(A_analytical)
print(f"\nAnalytical B:")
print(B_analytical)

diff_A = np.linalg.norm(A - A_analytical)
diff_B = np.linalg.norm(B - B_analytical)

print(f"\n||A_data - A_analytical|| = {diff_A:.6f}")
print(f"||B_data - B_analytical|| = {diff_B:.6f}")

# Test prediction accuracy
print(f"\n" + "=" * 60)
print("Testing Multi-Step Prediction Accuracy")
print("=" * 60)

env = gym.make(env_name)
obs, _ = env.reset(seed=42)
state_actual = obs['observation'][:4].copy()
state_predicted = state_actual.copy()

action = np.array([0.5, 0.3])
errors = []

for step in range(10):
    # Predict using fitted model
    state_predicted = A @ state_predicted + B @ action
    
    # Get actual next state
    obs, _, _, _, _ = env.step(action)
    state_actual = obs['observation'][:4]
    
    error = np.linalg.norm(state_predicted - state_actual)
    errors.append(error)
    
    print(f"Step {step}: error = {error:.6f}")

print(f"\nMean error: {np.mean(errors):.6f}")
print(f"Max error: {np.max(errors):.6f}")

env.close()

if np.mean(errors) < 0.01:
    print("\n✓ Data-driven dynamics are highly accurate!")
elif np.mean(errors) < 0.1:
    print(f"\n✓ Data-driven dynamics are reasonably accurate!")
else:
    print(f"\n⚠ High prediction error - check dynamics extraction")