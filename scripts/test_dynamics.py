"""
Test if PointMaze dynamics match our double integrator model.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

env = gym.make('PointMaze_UMaze-v3')

# Get dt
print("=" * 60)
print("PointMaze Dynamics Test")
print("=" * 60)

# Check environment parameters
if hasattr(env.unwrapped, 'dt'):
    print(f"Environment dt: {env.unwrapped.dt}")
elif hasattr(env.unwrapped, '_dt'):
    print(f"Environment dt: {env.unwrapped._dt}")
else:
    print("dt not found, trying to infer...")

# Reset
obs, _ = env.reset(seed=42)
state0 = obs['observation'][:4].copy()
print(f"\nInitial state: {state0}")
print(f"  Position: [{state0[0]:.4f}, {state0[1]:.4f}]")
print(f"  Velocity: [{state0[2]:.4f}, {state0[3]:.4f}]")

# Apply constant action
action = np.array([1.0, 0.0])  # Acceleration in x direction
print(f"\nAction: {action}")

obs, _, _, _, _ = env.step(action)
state1 = obs['observation'][:4].copy()
print(f"\nNext state: {state1}")
print(f"  Position: [{state1[0]:.4f}, {state1[1]:.4f}]")
print(f"  Velocity: [{state1[2]:.4f}, {state1[3]:.4f}]")

# Compute changes
delta_pos = state1[:2] - state0[:2]
delta_vel = state1[2:] - state0[2:]

print(f"\nChanges:")
print(f"  Δposition: {delta_pos}")
print(f"  Δvelocity: {delta_vel}")

# Test different dt values
print("\n" + "=" * 60)
print("Testing Double Integrator Fit")
print("=" * 60)

for dt in [0.01, 0.02, 0.05, 0.1]:
    # Predict using double integrator
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    B = np.array([
        [0.5 * dt**2, 0],
        [0, 0.5 * dt**2],
        [dt, 0],
        [0, dt]
    ])
    
    predicted = A @ state0 + B @ action
    error = np.linalg.norm(predicted - state1)
    
    print(f"\ndt = {dt}:")
    print(f"  Predicted: {predicted}")
    print(f"  Actual:    {state1}")
    print(f"  Error:     {error:.6f}")

# Multi-step test
print("\n" + "=" * 60)
print("Multi-Step Test (10 steps)")
print("=" * 60)

obs, _ = env.reset(seed=42)
state = obs['observation'][:4].copy()

# Best dt from above test
dt = 0.02  # You'll know after running

A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

B = np.array([
    [0.5 * dt**2, 0],
    [0, 0.5 * dt**2],
    [dt, 0],
    [0, dt]
])

action = np.array([1.0, 0.5])
errors = []

for step in range(10):
    # Predict
    predicted = A @ state + B @ action
    
    # Actual
    obs, _, _, _, _ = env.step(action)
    actual = obs['observation'][:4]
    
    error = np.linalg.norm(predicted - actual)
    errors.append(error)
    
    print(f"Step {step}: error = {error:.6f}")
    
    state = actual

print(f"\nMean error: {np.mean(errors):.6f}")
print(f"Max error: {np.max(errors):.6f}")

env.close()