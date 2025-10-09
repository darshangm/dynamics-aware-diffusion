# experiments/lqr/train_vanilla_only.py
#!/usr/bin/env python3
"""Train only the vanilla diffusion model."""

import os
import sys
import torch
import torch.optim as optim

sys.path.append('../..')

from src.data.data_loader import TrajectoryDataLoader
from src.models.vanilla_diffusion import VanillaDiffusion
from configs.base_config import LQRConfig

def main():
    """Train vanilla diffusion model only."""
    config = LQRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Vanilla Diffusion on {device}")
    
    # Setup data loader
    data_loader = TrajectoryDataLoader(config.train_data_dir, config.state_dim, config.control_dim)
    
    # Create method and model
    method = VanillaDiffusion()
    model = method.create_model(
        input_dim=config.model_config.input_dim,
        condition_dim=config.model_config.condition_dim,
        seq_len=config.model_config.seq_len
    ).to(device)
    
    # Create trainer
    trainer = method.create_trainer(model=model, timesteps=config.training_config.timesteps)
    
    # Setup optimizer and save directory
    optimizer = optim.Adam(model.parameters(), lr=config.training_config.learning_rate)
    save_dir = f"{config.get_save_dir('vanilla')}/models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    epochs = 100  # Start with just 100 epochs for testing
    losses = []
    
    for epoch in range(epochs):
        trajectories, conditions = data_loader.load_batch(config.training_config.batch_size)
        trajectories, conditions = trajectories.to(device), conditions.to(device)
        
        loss = trainer.train_step(trajectories, conditions, optimizer)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(save_dir, f"vanilla_diffusion_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)
            print(f"Model saved to {save_path}")
    
    print("Vanilla training complete!")

if __name__ == "__main__":
    main()

# experiments/lqr/train_model_based_only.py
#!/usr/bin/env python3
"""Train only the model-based diffusion model."""

import os
import sys
import torch
import torch.optim as optim

sys.path.append('../..')

from src.data.data_loader import TrajectoryDataLoader
from src.models.model_based_diffusion import ModelBasedDiffusion
from configs.base_config import LQRConfig

def main():
    """Train model-based diffusion model only."""
    config = LQRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Model-Based Diffusion on {device}")
    
    # Setup data loader
    data_loader = TrajectoryDataLoader(config.train_data_dir, config.state_dim, config.control_dim)
    
    # Create method and model
    method = ModelBasedDiffusion()
    model = method.create_model(
        input_dim=config.model_config.input_dim,
        condition_dim=config.model_config.condition_dim,
        seq_len=config.model_config.seq_len
    ).to(device)
    
    # Create trainer (includes dynamics matrices)
    trainer = method.create_trainer(
        model=model,
        state_dim=config.state_dim,
        control_dim=config.control_dim,
        seq_len=config.model_config.seq_len,
        timesteps=config.training_config.timesteps
    )
    
    # Setup optimizer and save directory
    optimizer = optim.Adam(model.parameters(), lr=config.training_config.learning_rate)
    save_dir = f"{config.get_save_dir('model_based')}/models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    epochs = 100  # Start with just 100 epochs for testing
    losses = []
    
    for epoch in range(epochs):
        trajectories, conditions = data_loader.load_batch(config.training_config.batch_size)
        trajectories, conditions = trajectories.to(device), conditions.to(device)
        
        loss = trainer.train_step(trajectories, conditions, optimizer)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(save_dir, f"model_based_diffusion_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)
            print(f"Model saved to {save_path}")
    
    print("Model-based training complete!")

if __name__ == "__main__":
    main()

# experiments/lqr/train_hankel_only.py
#!/usr/bin/env python3
"""Train only the Hankel diffusion model."""

import os
import sys
import torch
import torch.optim as optim

sys.path.append('../..')

from src.data.data_loader import TrajectoryDataLoader
from src.models.hankel_diffusion import HankelDiffusion
from configs.base_config import LQRConfig

def main():
    """Train Hankel diffusion model only."""
    config = LQRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Hankel Diffusion on {device}")
    
    # Check if Hankel data exists
    if not os.path.exists(config.hankel_data_dir):
        print(f"Hankel data directory not found: {config.hankel_data_dir}")
        print("Please ensure your Hankel training data is available.")
        return
    
    # Setup data loader
    data_loader = TrajectoryDataLoader(config.train_data_dir, config.state_dim, config.control_dim)
    
    # Create method and model
    method = HankelDiffusion()
    model = method.create_model(
        input_dim=config.model_config.input_dim,
        condition_dim=config.model_config.condition_dim,
        seq_len=config.model_config.seq_len
    ).to(device)
    
    # Create trainer (constructs Hankel matrix)
    print("Constructing Hankel matrix...")
    trainer = method.create_trainer(
        model=model,
        hankel_data_dir=config.hankel_data_dir,
        state_dim=config.state_dim,
        control_dim=config.control_dim,
        seq_len=config.model_config.seq_len,
        timesteps=config.training_config.timesteps,
        max_trajectories=100  # Use fewer trajectories for testing
    )
    
    # Setup optimizer and save directory
    optimizer = optim.Adam(model.parameters(), lr=config.training_config.learning_rate)
    save_dir = f"{config.get_save_dir('hankel')}/models"
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    epochs = 100  # Start with just 100 epochs for testing
    losses = []
    
    for epoch in range(epochs):
        trajectories, conditions = data_loader.load_batch(config.training_config.batch_size)
        trajectories, conditions = trajectories.to(device), conditions.to(device)
        
        loss = trainer.train_step(trajectories, conditions, optimizer)
        losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
        
        if (epoch + 1) % 50 == 0:
            save_path = os.path.join(save_dir, f"hankel_diffusion_model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_path)
            print(f"Model saved to {save_path}")
    
    print("Hankel training complete!")

if __name__ == "__main__":
    main()

# experiments/lqr/eval_single_method.py
#!/usr/bin/env python3
"""Evaluate a single trained method."""

import os
import sys
import torch
import numpy as np

sys.path.append('../..')

from src.data.data_loader import TrajectoryDataLoader
from src.models.vanilla_diffusion import load_vanilla_model
from src.models.model_based_diffusion import load_model_based_model
from src.models.hankel_diffusion import load_hankel_model
from configs.base_config import LQRConfig

def evaluate_method(method_name, model_epoch=100):
    """
    Evaluate a specific method.
    
    Args:
        method_name: "vanilla", "model_based", or "hankel"
        model_epoch: Epoch number of model to load
    """
    config = LQRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Evaluating {method_name} diffusion model (epoch {model_epoch})")
    
    # Load test data
    test_loader = TrajectoryDataLoader(config.test_data_dir, config.state_dim, config.control_dim)
    
    # Load model based on method
    model_path = f"{config.get_save_dir(method_name)}/models/{method_name}_diffusion_model_epoch_{model_epoch}.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    if method_name == "vanilla":
        model, trainer = load_vanilla_model(
            model_path, config.model_config.input_dim, 
            config.model_config.condition_dim, config.model_config.seq_len, device
        )
    elif method_name == "model_based":
        model, trainer = load_model_based_model(
            model_path, config.model_config.input_dim, config.model_config.condition_dim,
            config.model_config.seq_len, config.state_dim, config.control_dim, device
        )
    elif method_name == "hankel":
        model, trainer = load_hankel_model(
            model_path, config.model_config.input_dim, config.model_config.condition_dim,
            config.model_config.seq_len, config.hankel_data_dir, 
            config.state_dim, config.control_dim, device
        )
    else:
        print(f"Unknown method: {method_name}")
        return
    
    # Generate test trajectories
    test_trajectories, test_conditions, test_files = test_loader.load_specific(file_indices=list(range(10)))
    test_trajectories, test_conditions = test_trajectories.to(device), test_conditions.to(device)
    
    print(f"Generating {len(test_files)} trajectories...")
    
    with torch.no_grad():
        generated_samples = trainer.sample(
            (test_trajectories.shape[1], test_trajectories.shape[2]), 
            test_conditions
        )
    
    # Save results
    save_dir = f"{config.get_save_dir(method_name)}/test_results"
    os.makedirs(save_dir, exist_ok=True)
    
    for i, file_name in enumerate(test_files):
        file_base = os.path.splitext(file_name)[0]
        save_path = os.path.join(save_dir, f"{method_name}_{file_base}.npz")
        
        np.savez(
            save_path,
            generated=generated_samples[i].cpu().numpy(),
            real=test_trajectories[i].cpu().numpy(),
            condition=test_conditions[i].cpu().numpy()
        )
    
    print(f"Results saved to {save_dir}")

if __name__ == "__main__":
    # Example usage - change these as needed
    method = "vanilla"  # Change to "model_based" or "hankel"
    epoch = 100
    
    evaluate_method(method, epoch)