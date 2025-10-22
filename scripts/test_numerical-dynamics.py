"""Test trajectory-based dynamics fitting for PointMaze."""

import gymnasium as gym
import gymnasium_robotics
from m_diffuser.dynamics.extractor import TrajectoryDynamicsExtractor
import numpy as np

print("=" * 60)
print("Testing Trajectory-Based Dynamics Fitting")
print("=" * 60)

# Create extractor
extractor = TrajectoryDynamicsExtractor('PointMaze_UMaze-v3')

# Fit dynamics from trajectories
A, B = extractor.get_dynamics(num_trajectories=50, trajectory_length=100)

print(f"\nFitted A matrix:")
print(A)
print(f"\nFitted B matrix:")
print(B)

# Compare with analytical if available
print(f"\n" + "=" * 60)
print("Comparing with Analytical Dynamics")
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

print(f"Analytical A:")
print(A_analytical)
print(f"\nAnalytical B:")
print(B_analytical)

print(f"\nDifference in A: {np.linalg.norm(A - A_analytical):.6f}")
print(f"Difference in B: {np.linalg.norm(B - B_analytical):.6f}")

extractor.close()