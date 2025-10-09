import numpy as np


def construct_dynamics_matrices(state_dim, control_dim, seq_len, dt=0.1):
    """
    Construct system dynamics matrices for LQR double integrator system.
    This matches your original implementation exactly.
    
    Args:
        state_dim: Dimension of state (4 for double integrator)
        control_dim: Dimension of control (2 for double integrator)
        seq_len: Length of trajectory sequence
        dt: Discretization time step
        
    Returns:
        A_T: System dynamics matrix [seq_len*state_dim, state_dim]
        C_T: Control dynamics matrix [seq_len*state_dim, seq_len*control_dim]
    """
    
    # Double integrator continuous-time matrices
    # State: [x, y, vx, vy], Control: [ax, ay]
    Ac = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    Bc = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    
    # Discretize the system (exact discretization)
    A = np.eye(state_dim) + dt * Ac
    B = dt * Bc
    
    # Construct A_T matrix: stacked powers of A
    A_T = np.zeros((seq_len * state_dim, state_dim))
    for t in range(seq_len):
        A_T[t * state_dim:(t + 1) * state_dim, :] = np.linalg.matrix_power(A, t)
    
    # Construct C_T matrix: convolution matrix for controls
    C_T = np.zeros((seq_len * state_dim, seq_len * control_dim))
    for t in range(seq_len):
        for k in range(t):
            C_T[t * state_dim:(t + 1) * state_dim, k * control_dim:(k + 1) * control_dim] = \
                np.linalg.matrix_power(A, t - k - 1) @ B
    
    return A_T, C_T


def verify_dynamics_consistency(states, controls, A_T, C_T, initial_state, tolerance=1e-6):
    """
    Verify that a trajectory follows the system dynamics.
    
    Args:
        states: State trajectory [seq_len, state_dim]
        controls: Control trajectory [seq_len, control_dim]
        A_T: System dynamics matrix
        C_T: Control dynamics matrix
        initial_state: Initial state [state_dim]
        tolerance: Numerical tolerance for verification
        
    Returns:
        is_consistent: Boolean indicating if trajectory follows dynamics
        max_error: Maximum error across the trajectory
    """
    seq_len, state_dim = states.shape
    control_dim = controls.shape[1]
    
    # Flatten trajectories
    states_flat = states.reshape(-1)
    controls_flat = controls.reshape(-1)
    
    # Compute expected states using dynamics
    expected_states = A_T @ initial_state + C_T @ controls_flat
    
    # Compute error
    error = np.abs(expected_states - states_flat)
    max_error = np.max(error)
    
    is_consistent = max_error < tolerance
    
    return is_consistent, max_error


def create_lqr_system(state_dim=4, control_dim=2, dt=0.1):
    """
    Create the LQR system matrices for double integrator.
    
    Args:
        state_dim: State dimension (default 4)
        control_dim: Control dimension (default 2)
        dt: Discretization time step
        
    Returns:
        A: Discrete-time system matrix
        B: Discrete-time control matrix
        Q: State cost matrix
        R: Control cost matrix
    """
    # Continuous-time system
    Ac = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    Bc = np.array([
        [0, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    
    # Discretize
    A = np.eye(state_dim) + dt * Ac
    B = dt * Bc
    
    # Cost matrices (matching your original implementation)
    Q = np.diag([10, 10, 1, 1])  # Higher weight on position
    R = np.diag([1, 1])          # Control effort penalty
    
    return A, B, Q, R