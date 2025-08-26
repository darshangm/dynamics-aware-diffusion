#!/usr/bin/env python3
"""
Test script to verify that the restructured code works with your existing models and data.
Run this after setting up the new directory structure.
"""

import os
import torch
import sys
import numpy as np

# Add src to path
sys.path.append('.')

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data.data_loader import TrajectoryDataLoader
        from src.models.base_diffusion import BaseDiffusionModel, DiffusionMethod
        from src.models.vanilla_diffusion import VanillaDiffusion, load_vanilla_model
        from src.training.base_trainer import BaseDiffusionTrainer
        from configs.base_config import LQRConfig, ModelConfig, TrainingConfig
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading with new structure."""
    print("\nTesting data loading...")
    
    # Test with your existing data paths (adjust as needed)
    data_paths_to_test = [
        "./trajectories",  # Your original training data
        "./traj_test_set", # Your original test data
        "./traj_hankel",   # Your original hankel data
    ]
    
    for data_path in data_paths_to_test:
        if os.path.exists(data_path):
            try:
                from src.data.data_loader import TrajectoryDataLoader
                loader = TrajectoryDataLoader(data_path, state_dim=4, control_dim=2)
                
                # Test batch loading
                trajectories, conditions = loader.load_batch(batch_size=2)
                print(f"✓ {data_path}: Loaded batch - trajectories: {trajectories.shape}, conditions: {conditions.shape}")
                
                # Test specific loading
                trajs, conds, files = loader.load_specific(file_indices=[0, 1])
                print(f"✓ {data_path}: Loaded specific - {len(files)} files")
                
            except Exception as e:
                print(f"✗ {data_path}: Failed - {e}")
                return False
        else:
            print(f"⚠ {data_path}: Directory not found (skip if expected)")
    
    return True

def test_model_creation():
    """Test creating models with new structure."""
    print("\nTesting model creation...")
    
    try:
        from src.models.vanilla_diffusion import VanillaDiffusion
        from configs.base_config import LQRConfig
        
        config = LQRConfig()
        method = VanillaDiffusion()
        
        # Create model
        model = method.create_model(
            input_dim=config.model_config.input_dim,
            condition_dim=config.model_config.condition_dim,
            seq_len=config.model_config.seq_len,
            time_dim=config.model_config.time_dim,
            hidden_dim=config.model_config.hidden_dim,
            latent_dim=config.model_config.latent_dim
        )
        
        # Create trainer
        trainer = method.create_trainer(
            model=model,
            timesteps=config.training_config.timesteps,
            beta_start=config.training_config.beta_start,
            beta_end=config.training_config.beta_end
        )
        
        print(f"✓ Model created: {model.__class__.__name__}")
        print(f"✓ Trainer created: {trainer.__class__.__name__}")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        batch_size = 2
        seq_len = config.model_config.seq_len
        feature_dim = config.model_config.input_dim
        condition_dim = config.model_config.condition_dim
        
        x = torch.randn(batch_size, seq_len, feature_dim, device=device)
        t = torch.randint(0, 100, (batch_size,), device=device)
        condition = torch.randn(batch_size, condition_dim, device=device)
        
        with torch.no_grad():
            output = model(x, t, condition)
        
        print(f"✓ Forward pass successful: input {x.shape} -> output {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_existing_model_loading():
    """Test loading your existing saved models."""
    print("\nTesting existing model loading...")
    
    # Common paths where your models might be saved
    model_paths_to_test = [
        "./saved_models/diffusion_model_epoch_35500.pt",
        "./saved_models/diffusion_model_final.pt",
        "./saved_models_hankel/hankel_diffusion_model_epoch_20000.pt",
    ]
    
    for model_path in model_paths_to_test:
        if os.path.exists(model_path):
            try:
                from src.models.vanilla_diffusion import load_vanilla_model
                
                model, trainer = load_vanilla_model(
                    model_path=model_path,
                    input_dim=6,  # 4 + 2
                    condition_dim=8,  # 2 * 4
                    seq_len=30
                )
                
                print(f"✓ {model_path}: Model loaded successfully")
                
                # Test that model can run inference
                device = next(model.parameters()).device
                x = torch.randn(1, 30, 6, device=device)
                t = torch.zeros(1, device=device, dtype=torch.long)
                condition = torch.randn(1, 8, device=device)
                
                with torch.no_grad():
                    output = model(x, t, condition)
                
                print(f"✓ {model_path}: Inference test passed")
                
            except Exception as e:
                print(f"✗ {model_path}: Failed to load - {e}")
                return False
        else:
            print(f"⚠ {model_path}: File not found (skip if expected)")
    
    return True

def test_config_system():
    """Test the configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from configs.base_config import LQRConfig, ComplexTaskConfig
        
        # Test LQR config
        lqr_config = LQRConfig()
        print(f"✓ LQR config created: {lqr_config.name}")
        print(f"  - State dim: {lqr_config.state_dim}")
        print(f"  - Control dim: {lqr_config.control_dim}")
        print(f"  - Model input dim: {lqr_config.model_config.input_dim}")
        print(f"  - Train data dir: {lqr_config.train_data_dir}")
        
        # Test complex task config
        complex_config = ComplexTaskConfig()
        print(f"✓ Complex task config created: {complex_config.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config system failed: {e}")
        return False

def test_directory_structure():
    """Test that expected directories exist or can be created."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "src/data",
        "src/models", 
        "src/training",
        "configs"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}: exists")
        else:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✓ {dir_path}: created")
            except Exception as e:
                print(f"✗ {dir_path}: failed to create - {e}")
                return False
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("MIGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Imports", test_imports),
        ("Configuration System", test_config_system),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Existing Model Loading", test_existing_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}: Crashed - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status:4} | {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Your migration is ready.")
        print("\nNext steps:")
        print("1. Move your data to the new directory structure")
        print("2. Try running the example training script") 
        print("3. Implement model-based and Hankel diffusion methods")
    else:
        print(f"\n⚠ {total_tests - total_passed} tests failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main()