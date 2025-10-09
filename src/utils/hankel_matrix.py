import os
import torch
import numpy as np
from tqdm import tqdm


def construct_hankel_matrix(data_dir, state_dim=4, control_dim=2, seq_len=30, max_trajectories=None):
    """
    Construct a Hankel matrix from trajectory data.
    This matches your original implementation exactly.
    
    Args:
        data_dir: Directory containing trajectory data files (.npz)
        state_dim: Dimension of state vectors
        control_dim: Dimension of control vectors
        seq_len: Expected sequence length for subsequences
        max_trajectories: Maximum number of trajectories to include (None = use all)
        
    Returns:
        H: Hankel matrix as torch tensor
        proj_matrix: Projection matrix P = U*U^T from SVD
        feature_dim: Total dimension of state+control vectors
    """
    
    print(f"Constructing Hankel matrix from trajectories in {data_dir}")
    
    # Get list of npz files
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    
    if len(file_list) == 0:
        raise ValueError(f"No .npz files found in {data_dir}")
    
    if max_trajectories is not None:
        file_list = file_list[:min(max_trajectories, len(file_list))]
    
    print(f"Using {len(file_list)} trajectory files")
    
    # Process each trajectory file
    trajectory_columns = []
    feature_dim = state_dim + control_dim
    
    for file_name in tqdm(file_list, desc="Processing trajectory files"):
        try:
            file_path = os.path.join(data_dir, file_name)
            data = np.load(file_path)
            
            # Extract states and controls
            states = data['states'][:-1]  # Remove the last state to match control length
            controls = data['controls']
            
            # Verify data dimensions
            if states.shape[1] != state_dim:
                print(f"Warning: State dimension mismatch in {file_name}. Expected {state_dim}, got {states.shape[1]}")
                continue
                
            if controls.shape[1] != control_dim:
                print(f"Warning: Control dimension mismatch in {file_name}. Expected {control_dim}, got {controls.shape[1]}")
                continue
            
            # Check if trajectory length is sufficient
            traj_len = min(states.shape[0], controls.shape[0])
            if traj_len < seq_len:
                print(f"Warning: Trajectory too short in {file_name}. Expected at least {seq_len}, got {traj_len}")
                continue
                
            # Create multiple columns from this trajectory using sliding window
            for start_idx in range(traj_len - seq_len + 1):
                # Extract subsequence
                states_subseq = states[start_idx:start_idx+seq_len]
                controls_subseq = controls[start_idx:start_idx+seq_len]
                
                # Create a column vector by flattening and concatenating
                states_flat = states_subseq.reshape(-1)  # [seq_len*state_dim]
                controls_flat = controls_subseq.reshape(-1)  # [seq_len*control_dim]
                
                # Combine into a single column
                traj_column = np.concatenate([states_flat, controls_flat])
                trajectory_columns.append(traj_column)
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue
    
    if not trajectory_columns:
        raise ValueError("No valid trajectories could be processed")
    
    # Stack columns to form the Hankel matrix
    H = np.column_stack(trajectory_columns)
    print(f"Hankel matrix shape: {H.shape}")
    
    # Convert to torch tensor
    H_torch = torch.tensor(H, dtype=torch.float32)
    
    # Compute SVD for projection
    print("Computing SVD of Hankel matrix...")
    U, S, V = torch.svd(H_torch)
    
    # Print singular values to assess numerical rank
    print(f"Top 10 singular values: {S[:10].cpu().numpy()}")
    
    # Compute projection matrix P = U*U^T
    proj_matrix = torch.matmul(U, U.t())
    
    return H_torch, proj_matrix, feature_dim


def project_onto_hankel(trajectory_batch, proj_matrix, seq_len, feature_dim):
    """
    Project a batch of trajectories onto the column space of the Hankel matrix.
    
    Args:
        trajectory_batch: Batch of trajectories [batch_size, seq_len, feature_dim]
        proj_matrix: Projection matrix from Hankel SVD
        seq_len: Sequence length
        feature_dim: Feature dimension (state_dim + control_dim)
        
    Returns:
        Projected trajectories with the same shape
    """
    batch_size = trajectory_batch.shape[0]
    
    # Verify input shapes
    if trajectory_batch.shape[1] != seq_len or trajectory_batch.shape[2] != feature_dim:
        raise ValueError(f"Expected trajectory shape [batch_size, {seq_len}, {feature_dim}], "
                         f"got {trajectory_batch.shape}")
    
    # Reshape trajectories to match Hankel column format
    # [batch_size, seq_len, feature_dim] -> [batch_size, seq_len*feature_dim]
    traj_flat = trajectory_batch.reshape(batch_size, -1)
    
    # Verify projection matrix shape compatibility
    expected_proj_dim = seq_len * feature_dim
    if proj_matrix.shape[0] != expected_proj_dim or proj_matrix.shape[1] != expected_proj_dim:
        raise ValueError(f"Projection matrix has shape {proj_matrix.shape}, "
                         f"expected [{expected_proj_dim}, {expected_proj_dim}]")
    
    # Project each trajectory
    projected_flat = torch.matmul(proj_matrix, traj_flat.t()).t()
    
    # Reshape back to original format
    projected = projected_flat.reshape(batch_size, seq_len, feature_dim)
    
    return projected


def analyze_hankel_rank(H, threshold=1e-6):
    """
    Analyze the numerical rank of the Hankel matrix.
    
    Args:
        H: Hankel matrix [rows, cols]
        threshold: Threshold for determining numerical rank
        
    Returns:
        numerical_rank: Numerical rank of the matrix
        singular_values: All singular values
        explained_variance_ratio: Ratio of variance explained by each component
    """
    U, S, V = torch.svd(H)
    
    # Compute numerical rank
    numerical_rank = torch.sum(S > threshold).item()
    
    # Compute explained variance ratio
    total_variance = torch.sum(S ** 2)
    explained_variance_ratio = (S ** 2) / total_variance
    
    return numerical_rank, S, explained_variance_ratio


def save_hankel_data(H, proj_matrix, save_path):
    """
    Save Hankel matrix and projection matrix for later use.
    
    Args:
        H: Hankel matrix
        proj_matrix: Projection matrix
        save_path: Path to save the data
    """
    torch.save({
        'hankel_matrix': H,
        'projection_matrix': proj_matrix
    }, save_path)
    print(f"Hankel data saved to {save_path}")


def load_hankel_data(save_path):
    """
    Load precomputed Hankel matrix and projection matrix.
    
    Args:
        save_path: Path to the saved data
        
    Returns:
        H: Hankel matrix
        proj_matrix: Projection matrix
    """
    data = torch.load(save_path)
    return data['hankel_matrix'], data['projection_matrix']