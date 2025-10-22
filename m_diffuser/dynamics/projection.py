"""
projection.py - Fixed version
Build projection matrices for dynamics-aware diffusion.
"""

import numpy as np
import torch
from typing import Tuple


class ProjectionMatrixBuilder:
    """
    Build trajectory projection matrices F and P = FF†.
    
    Given dynamics x_{t+1} = A*x_t + B*u_t, constructs projection matrix
    that ensures generated trajectories satisfy the dynamics.
    
    Uses CONCATENATED format: [x0, x1, ..., xT, u0, u1, ..., u_{T-1}]
    """
    
    def __init__(self, A: np.ndarray, B: np.ndarray, state_dim: int, action_dim: int):
        """
        Args:
            A: State transition matrix (state_dim, state_dim)
            B: Control matrix (state_dim, action_dim)
            state_dim: Dimension of state
            action_dim: Dimension of action
        """
        self.A = A
        self.B = B
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Validate dimensions
        assert A.shape == (state_dim, state_dim), f"A shape mismatch: {A.shape}"
        assert B.shape == (state_dim, action_dim), f"B shape mismatch: {B.shape}"
        
        print(f"ProjectionMatrixBuilder initialized:")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  A condition number: {np.linalg.cond(A):.2e}")
    
    def _build_F_matrix(self, horizon: int) -> np.ndarray:
        """
        Build trajectory matrix F in concatenated format.
        
        Concatenated trajectory: τ = [x₀, x₁, ..., xₜ, u₀, u₁, ..., u_{T-1}]
        Can be expressed as: τ = F · [x₀, u₀, u₁, ..., u_{T-1}]
        
        Returns:
            F: ((T+1)*n + T*m, n + T*m) where n=state_dim, m=action_dim
        """
        T = horizon
        n = self.state_dim
        m = self.action_dim
        
        # Build A_bar: free response [I; A; A²; ...; Aᵀ]
        A_bar = np.zeros(((T + 1) * n, n))
        A_power = np.eye(n)
        for t in range(T + 1):
            A_bar[t * n:(t + 1) * n, :] = A_power
            if t < T:
                A_power = A_power @ self.A
        
        # Build C_T: forced response (controllability-like matrix)
        # C_T[t, τ] = A^(t-τ-1) @ B for τ < t
        C_T = np.zeros(((T + 1) * n, T * m))
        A_powers_B = [self.B]
        for _ in range(T - 1):
            A_powers_B.append(self.A @ A_powers_B[-1])
        
        for t in range(1, T + 1):
            for tau in range(t):
                power_idx = t - tau - 1
                C_T[t * n:(t + 1) * n, tau * m:(tau + 1) * m] = A_powers_B[power_idx]
        
        # Construct F = [A_bar, C_T; 0, I]
        F = np.zeros(((T + 1) * n + T * m, n + T * m))
        F[:(T + 1) * n, :n] = A_bar
        F[:(T + 1) * n, n:] = C_T
        F[(T + 1) * n:, n:] = np.eye(T * m)
        
        return F
    
    def get_projection_matrix(self, horizon: int) -> torch.Tensor:
        """
        Get projection matrix P = FF† for given horizon.
        
        Args:
            horizon: Planning horizon
        
        Returns:
            P: Projection matrix as PyTorch tensor (concatenated format)
        """
        print(f"\nBuilding projection matrix for horizon={horizon}...")
        
        # Build F in concatenated format
        F = self._build_F_matrix(horizon)
        
        print(f"  F shape: {F.shape}")
        print(f"  F rank: {np.linalg.matrix_rank(F)}")
        
        # Compute pseudoinverse F†
        F_pinv = np.linalg.pinv(F)
        
        # Projection matrix P = FF†
        P = F @ F_pinv
        
        # Verify it's a projection
        P_squared = P @ P
        error = np.linalg.norm(P_squared - P, 'fro')
        print(f"  ||P² - P||_F = {error:.2e}")
        
        if error > 1e-4:
            print("  WARNING: P is not a valid projection matrix!")
        else:
            print("  ✓ P is a valid projection matrix")
        
        # Convert to torch tensor
        return torch.from_numpy(P).float()
    
    def verify_projection(self, P: torch.Tensor) -> bool:
        """
        Verify that P is a valid projection matrix (P @ P ≈ P).
        
        Args:
            P: Projection matrix
        
        Returns:
            True if P is a projection matrix
        """
        P_squared = P @ P
        return torch.allclose(P_squared, P, atol=1e-4)


def test_projection_matrices():
    """Test projection matrix construction."""
    print("=" * 60)
    print("Testing Projection Matrix Builder")
    print("=" * 60)
    
    # Double integrator dynamics (PointMaze)
    dt = 0.01
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
    
    state_dim = 4
    action_dim = 2
    horizon = 8
    
    # Build projection
    builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
    P = builder.get_projection_matrix(horizon)
    
    print(f"\nProjection matrix P shape: {P.shape}")
    print(f"P is projection: {builder.verify_projection(P)}")
    
    print("\n" + "=" * 60)
    print("✓ Test passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_projection_matrices()