"""
Build projection matrices for dynamics-aware diffusion.

Implements the F and P matrices from:
"Dynamics-aware Diffusion Models for Planning and Control"
"""

import numpy as np
import torch
from typing import Tuple


class ProjectionMatrixBuilder:
    """
    Build trajectory projection matrices F and P.
    
    Given dynamics x_{t+1} = A*x_t + B*u_t, constructs:
    - F: Trajectory matrix such that τ = F * [x_0; u_{0:T-1}]
    - P = FF†: Projection onto feasible trajectory space
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
        
        # Cache for different horizons
        self._F_cache = {}
        self._P_cache = {}
    
    def build_F_matrix(self, horizon: int) -> np.ndarray:
        """
        Build trajectory matrix F for given horizon.
        
        Trajectory representation:
        τ = [x(0:T); u(0:T-1)] = F * [x(0); u(0:T-1)]
        
        Where F = [A_bar, C_T; 0, I] with:
        - A_bar: Free response matrix [I; A; A²; ...; A^T]
        - C_T: Forced response matrix (controllability-like)
        
        Args:
            horizon: Planning horizon T
        
        Returns:
            F: Trajectory matrix ((T+1)*n + T*m, n + T*m)
        """
        if horizon in self._F_cache:
            return self._F_cache[horizon]
        
        T = horizon
        n = self.state_dim
        m = self.action_dim
        
        # Build A_bar (free response matrix)
        A_bar = self._build_free_response(T)
        
        # Build C_T (forced response matrix)
        C_T = self._build_forced_response(T)
        
        # Construct F = [A_bar, C_T; 0, I]
        F = np.zeros(((T + 1) * n + T * m, n + T * m))
        
        # Top block: state equations
        F[:(T + 1) * n, :n] = A_bar
        F[:(T + 1) * n, n:] = C_T
        
        # Bottom block: control inputs (identity)
        F[(T + 1) * n:, n:] = np.eye(T * m)
        
        self._F_cache[horizon] = F
        return F
    
    def _build_free_response(self, T: int) -> np.ndarray:
        """
        Build free response matrix A_bar = [I; A; A²; ...; A^T].
        
        Args:
            T: Horizon length
        
        Returns:
            A_bar: ((T+1)*n, n) matrix
        """
        n = self.state_dim
        A_bar = np.zeros(((T + 1) * n, n))
        
        A_power = np.eye(n)
        for t in range(T + 1):
            A_bar[t * n:(t + 1) * n, :] = A_power
            if t < T:  # Don't compute unnecessary matrix power
                A_power = A_power @ self.A
        
        return A_bar
    
    def _build_forced_response(self, T: int) -> np.ndarray:
        """
        Build forced response matrix C_T.
        
        C_T[t, tau] = A^{t-tau-1} B for tau < t, 0 otherwise
        
        Structure:
        [0,        0,      ..., 0      ]
        [B,        0,      ..., 0      ]
        [AB,       B,      ..., 0      ]
        [...,      ...,    ..., ...    ]
        [A^{T-1}B, A^{T-2}B, ..., B    ]
        
        Args:
            T: Horizon length
        
        Returns:
            C_T: ((T+1)*n, T*m) matrix
        """
        n = self.state_dim
        m = self.action_dim
        C_T = np.zeros(((T + 1) * n, T * m))
        
        # Precompute powers of A times B
        A_powers_B = [self.B]  # A^0 * B = B
        for _ in range(T - 1):
            A_powers_B.append(self.A @ A_powers_B[-1])
        
        # Fill C_T
        for t in range(1, T + 1):  # Start from t=1 (t=0 has no control history)
            for tau in range(t):
                power_idx = t - tau - 1
                C_T[t * n:(t + 1) * n, tau * m:(tau + 1) * m] = A_powers_B[power_idx]
        
        return C_T
    
    def get_projection_matrix(self, horizon: int) -> torch.Tensor:
        """
        Get projection matrix P = FF† for given horizon.
        
        Args:
            horizon: Planning horizon
        
        Returns:
            P: Projection matrix as PyTorch tensor
        """
        if horizon in self._P_cache:
            return self._P_cache[horizon]
        
        # Build F
        F = self.build_F_matrix(horizon)
        
        # Compute pseudoinverse F†
        F_pinv = np.linalg.pinv(F)
        
        # Projection matrix P = FF†
        P = F @ F_pinv
        
        # Convert to torch tensor
        P_torch = torch.from_numpy(P).float()
        
        self._P_cache[horizon] = P_torch
        return P_torch
    
    def project_trajectory(self, trajectory: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Project trajectory onto feasible space.
        
        Args:
            trajectory: (batch, horizon, state_dim + action_dim)
            horizon: Planning horizon
        
        Returns:
            projected_trajectory: Feasible trajectory
        """
        batch_size = trajectory.shape[0]
        
        # Get projection matrix
        P = self.get_projection_matrix(horizon).to(trajectory.device)
        
        # Flatten trajectory
        traj_flat = trajectory.reshape(batch_size, -1)
        
        # Project: τ' = P * τ
        traj_proj = (P @ traj_flat.T).T
        
        # Reshape back
        return traj_proj.reshape(trajectory.shape)


def test_projection_matrices():
    """Test projection matrix construction."""
    print("Testing Projection Matrix Builder...")
    
    # Double integrator dynamics
    dt = 0.1
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
    horizon = 10
    
    # Build projection
    builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
    F = builder.build_F_matrix(horizon)
    P = builder.get_projection_matrix(horizon)
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Horizon: {horizon}")
    print(f"F shape: {F.shape}")
    print(f"P shape: {P.shape}")
    print(f"P is projection: {torch.allclose(P @ P, P, atol=1e-5)}")
    
    # Test projection
    traj = torch.randn(1, horizon, state_dim + action_dim)
    traj_proj = builder.project_trajectory(traj, horizon)
    print(f"Trajectory shape: {traj.shape}")
    print(f"Projected trajectory shape: {traj_proj.shape}")
    
    print("✓ Projection matrix test passed!")


if __name__ == "__main__":
    test_projection_matrices()