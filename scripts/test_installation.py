"""
Test script to verify installation and setup.
Run this to check if all dependencies are properly installed.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'gymnasium': 'Gymnasium',
        'gymnasium_robotics': 'Gymnasium-Robotics',
        'minari': 'Minari',
        'mujoco': 'MuJoCo',
        'einops': 'Einops',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
    }
    
    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_cuda():
    """Test CUDA availability."""
    import torch
    
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ CUDA not available - will use CPU")
    
    return True


def test_environment():
    """Test creating a Gymnasium environment."""
    import gymnasium as gym
    
    print("\nTesting Gymnasium environment...")
    try:
        env = gym.make('HalfCheetah-v5')
        obs, info = env.reset()
        print(f"  ✓ Created HalfCheetah-v5")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        env.close()
        return True
    except Exception as e:
        print(f"  ✗ Failed to create environment: {e}")
        return False


def test_model():
    """Test creating model components."""
    print("\nTesting model creation...")
    
    try:
        from m_diffuser.models.temporal_unet import TemporalUnet
        from m_diffuser.models.diffusion import GaussianDiffusion
        
        # Create model
        unet = TemporalUnet(
            transition_dim=23,
            dim=64,
            dim_mults=(1, 2, 4)
        )
        
        diffusion = GaussianDiffusion(
            model=unet,
            horizon=32,
            observation_dim=17,
            action_dim=6,
            n_timesteps=100
        )
        
        print("  ✓ Created Temporal U-Net")
        print("  ✓ Created Diffusion model")
        
        # Test forward pass
        import torch
        x = torch.randn(2, 32, 23)
        loss = diffusion(x)
        print(f"  ✓ Forward pass successful (loss: {loss.item():.4f})")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minari():
    """Test Minari dataset availability."""
    import minari
    
    print("\nTesting Minari...")
    try:
        datasets = minari.list_remote_datasets()
        print(f"  ✓ Found {len(datasets)} remote datasets")
        
        # Check if any locomotion datasets are available
        locomotion = [d for d in datasets if any(env in d for env in ['halfcheetah', 'hopper', 'walker'])]
        if locomotion:
            print(f"  ✓ Locomotion datasets available: {len(locomotion)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Minari test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Modern Diffuser - Installation Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("Environment", test_environment()))
    results.append(("Model", test_model()))
    results.append(("Minari", test_minari()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Download datasets: python scripts/download_data.py")
        print("  2. Start training: python scripts/train.py --help")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())