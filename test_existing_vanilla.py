#!/usr/bin/env python3
"""
Test script to load your existing vanilla model and generate some trajectories
using the new restructured code.
"""

import os
import sys
import torch
import numpy as np

# Add src to path
sys.path.append('.')

from src.data.data_loader import TrajectoryDataLoader
from src.models.vanilla_diffusion import load_vanilla_model
from configs.base_config import LQRConfig

def test_existing_vanilla_model():
    """Test loading and using your existing vanilla diffusion model."""
    
    config = LQRConfig()
    
    # Update these paths to match where you moved your files
    model_path = "./results/lqr/vanilla/models/diffusion_model_epoch_35500.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please update the model_path in this script to point to your actual model file")
        return False
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model
        model, trainer = load_vanilla_model(
            model_path=model_path,
            input_dim=config.model_config.input_dim,  # 6
            condition_dim=config.model_config.condition_dim,  # 8
            seq_len=config.model_config.seq_len  # 30
        )
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return False
    
    # Test data loading
    test_data_path = "./data/lqr/test"
    if not os.path.exists(test_data_path):
        print(f"Test data not found at {test_data_path}")
        print("Please move your traj_test_set data to ./data/lqr/test/")
        return False
    
    try:
        # Load test data
        data_loader = TrajectoryDataLoader(test_data_path, config.state_dim, config.control_dim)
        
        # Load a few test trajectories
        test_trajectories, test_conditions, test_files = data_loader.load_specific(file_indices=[0, 1, 2])
        print(f"Loaded {len(test_files)} test trajectories")
        
        # Generate trajectories with the model
        device = next(model.parameters()).device
        test_trajectories = test_trajectories.to(device)
        test_conditions = test_conditions.to(device)
        
        print("Generating trajectories...")
        
        with torch.no_grad():
            seq_len = test_trajectories.shape[1]
            feature_dim = test_trajectories.shape[2]
            generated_samples = trainer.sample((seq_len, feature_dim), test_conditions)
        
        print(f"Generated trajectories shape: {generated_samples.shape}")
        
        # Save a sample result to verify it works
        save_dir = "./results/lqr/vanilla/test_output"
        os.makedirs(save_dir, exist_ok=True)
        
        gen_np = generated_samples.cpu().numpy()
        real_np = test_trajectories.cpu().numpy()
        cond_np = test_conditions.cpu().numpy()
        
        # Save first trajectory as an example
        file_base = os.path.splitext(test_files[0])[0]
        
        gen_states = gen_np[0, :, :config.state_dim]
        gen_controls = gen_np[0, :, config.state_dim:]
        real_states = real_np[0, :, :config.state_dim]  
        real_controls = real_np[0, :, config.state_dim:]
        
        initial_state = cond_np[0, :config.state_dim]
        target_state = cond_np[0, config.state_dim:2*config.state_dim]
        
        save_path = os.path.join(save_dir, f"vanilla_test_{file_base}.npz")
        np.savez(
            save_path,
            gen_states=gen_states,
            gen_controls=gen_controls,
            real_states=real_states,
            real_controls=real_controls,
            initial_state=initial_state,
            target_state=target_state,
            original_file=test_files[0]
        )
        
        print(f"Test trajectory saved to: {save_path}")
        print("SUCCESS: Your existing model works with the new code structure!")
        return True
        
    except Exception as e:
        print(f"Error during trajectory generation: {e}")
        return False

if __name__ == "__main__":
    print("Testing existing vanilla model with new code structure...")
    print("=" * 60)
    
    success = test_existing_vanilla_model()
    
    if success:
        print("\n" + "=" * 60)
        print("MIGRATION SUCCESSFUL!")
        print("Your existing vanilla model works perfectly with the new structure.")
        print("\nNext steps:")
        print("1. Implement model-based diffusion trainer")  
        print("2. Implement Hankel diffusion trainer")
        print("3. Create unified evaluation scripts")
    else:
        print("\nPlease fix the issues above before proceeding.")