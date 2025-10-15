"""
Build projection matrices for dynamics-aware diffusion.

Implements the F and P matrices with direct interleaved format support.
"""

import numpy as np
import torch
from typing import Tuple


class ProjectionMatrixBuilder:
    """
    Build trajectory projection matrices F and P = FF†.
    
    Given dynamics x_{t+1} = A*x_t + B*u_t, constructs projection matrix
    that ensures generated trajectories satisfy the dynamics.
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
    
    def _build_F_matrix(self, horizon: int) -> np.ndarray:
        """
        Build trajectory matrix F in concatenated format.
        
        Concatenated trajectory: τ = [x₀, x₁, ..., xₜ, u₀, u₁, ..., uₜ₋₁]
        Can be expressed as: τ = F · [x₀, u₀, u₁, ..., uₜ₋₁]
        
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
    
    def _build_permutation_matrix(self, horizon: int) -> np.ndarray:
        """
        Build permutation matrix Q: interleaved → concatenated.
        
        Interleaved: [s₀, a₀, s₁, a₁, ..., sₜ]
        Concatenated: [s₀, s₁, ..., sₜ, a₀, a₁, ..., aₜ₋₁]
        
        Returns:
            Q: Permutation matrix such that τ_concat = Q @ τ_interleaved
        """
        T = horizon
        n = self.state_dim
        m = self.action_dim
        
        n_concat = (T + 1) * n + T * m
        n_interleaved = T * (n + m) + n
        
        Q = np.zeros((n_concat, n_interleaved))
        
        concat_idx = 0
        
        # States section in concatenated
        for t in range(T + 1):
            for d in range(n):
                if t < T:
                    # State at timestep t in interleaved format
                    interleaved_idx = t * (n + m) + d
                else:
                    # Last state (no action after it)
                    interleaved_idx = T * (n + m) + d
                
                Q[concat_idx, interleaved_idx] = 1.0
                concat_idx += 1
        
        # Actions section in concatenated
        for t in range(T):
            for d in range(m):
                # Action at timestep t (after state in interleaved)
                interleaved_idx = t * (n + m) + n + d
                Q[concat_idx, interleaved_idx] = 1.0
                concat_idx += 1
        
        return Q
    
    def get_projection_matrix(self, horizon: int, interleaved: bool = True) -> torch.Tensor:
        """
        Get projection matrix P = FF† for given horizon.
        
        Args:
            horizon: Planning horizon
            interleaved: If True, return P for interleaved format (default: True)
                        If False, return P for concatenated format
        
        Returns:
            P: Projection matrix as PyTorch tensor
        """
        # Build F in concatenated format
        F = self._build_F_matrix(horizon)
        
        # Compute pseudoinverse F†
        F_pinv = np.linalg.pinv(F)
        
        # Projection matrix P = FF†
        P_concat = F @ F_pinv
        
        if interleaved:
            # Transform to interleaved format: P_I = Q^T @ P_C @ Q
            Q = self._build_permutation_matrix(horizon)
            P = Q.T @ P_concat @ Q
        else:
            P = P_concat
        
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
    """Test projection matrix construction and interleaved format."""
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
    horizon = 16
    
    # Build projections
    builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
    
    print(f"\nSystem:")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print(f"  Horizon: {horizon}")
    
    # Concatenated format
    P_concat = builder.get_projection_matrix(horizon, interleaved=False)
    print(f"\nConcatenated format:")
    print(f"  P shape: {P_concat.shape}")
    print(f"  P is projection: {builder.verify_projection(P_concat)}")
    
    # Interleaved format
    P_interleaved = builder.get_projection_matrix(horizon, interleaved=True)
    print(f"\nInterleaved format:")
    print(f"  P shape: {P_interleaved.shape}")
    print(f"  P is projection: {builder.verify_projection(P_interleaved)}")
    
    # Test equivalence
    print("\n" + "=" * 60)
    print("Testing Equivalence of Both Formats")
    print("=" * 60)
    
    # Create random trajectory in interleaved format
    transition_dim = state_dim + action_dim
    traj_interleaved = torch.randn(horizon * transition_dim + state_dim)
    
    # Method 1: Convert to concatenated, project, convert back
    Q = builder._build_permutation_matrix(horizon)
    Q_torch = torch.from_numpy(Q).float()
    
    traj_concat = Q_torch @ traj_interleaved
    traj_proj_concat = P_concat @ traj_concat
    result_1 = Q_torch.T @ traj_proj_concat
    
    # Method 2: Direct projection in interleaved format
    result_2 = P_interleaved @ traj_interleaved
    
    # Compare
    error = torch.norm(result_1 - result_2).item()
    print(f"\nError between methods: {error:.10f}")
    
    if error < 1e-8:
        print("✓ Both methods are mathematically equivalent!")
    else:
        print("✗ Methods don't match!")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    test_projection_matrices()