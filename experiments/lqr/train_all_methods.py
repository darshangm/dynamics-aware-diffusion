#!/usr/bin/env python3
"""
Unified training script for all three diffusion methods:
- Vanilla diffusion (no constraints)
- Model-based diffusion (physics constraints) 
- Hankel diffusion (data-driven constraints)

This demonstrates how clean the new structure makes training multiple methods.
"""

import os
import sys
import torch
import torch.optim as optim

# Add src to path
sys.path.append('../..')

from src.data.data_loader import TrajectoryDataLoader
from src.models.vanilla_diffusion import VanillaDiffusion
from src.models.model_based_diffusion import ModelBasedDiffusion
from src.models.hankel_diffusion import HankelDiffusion
from configs.base_config import LQRConfig


def train_method(method_class, method_name, config, device, epochs=1000):
    """
    Train a specific diffusion method.
    
    Args:
        method_class: Class implementing the diffusion method
        method_name: Name of the method for logging
        config: Experiment configuration
        device: Device to train on
        epochs: Number of training epochs
        
    Returns:
        model: Trained model
        trainer: Trained trainer
        losses: Training losses
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {method_name.upper()} DIFFUSION")
    print(f"{'='*60}")
    
    # Setup data loader
    data_loader = TrajectoryDataLoader(
        config.train_data_dir, 
        config.state_dim, 
        config.control_dim
    )
    
    # Create method instance
    method = method_class()
    
    # Create model
    model = method.create_model(
        input_dim=config.model_config.input_dim,
        condition_dim=config.model_config.condition_dim,
        seq_len=config.model_config.seq_len,
        time_dim=config.model_config.time_dim,
        hidden_dim=config.model_config.hidden_dim,
        latent_dim=config.model_config.latent_dim
    ).to(device)
    
    # Create trainer (method-specific parameters)
    if method_name == "vanilla":
        trainer = method.create_trainer(
            model=model,
            timesteps=config.training_config.timesteps,
            beta_start=config.training_config.beta_start,
            beta_end=config.training_config.beta_end
        )
    elif method_name == "model_based":
        trainer = method.create_trainer(
            model=model,
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            seq_len=config.model_config.seq_len,
            timesteps=config.training_config.timesteps,
            beta_start=config.training_config.beta_start,
            beta_end=config.training_config.beta_end
        )
    elif method_name == "hankel":
        trainer = method.create_trainer(
            model=model,
            hankel_data_dir=config.hankel_data_dir,
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            seq_len=config.model_config.seq_len,
            timesteps=config.training_config.timesteps,
            beta_start=config.training_config.beta_start,
            beta_end=config.training_config.beta_end,
            max_trajectories=500
        )
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training_config.learning_rate)
    
    # Create save directory
    save_dir = config.get_save_dir(method_name)
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    
    # Training loop
    losses = []
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Load batch
        trajectories, conditions = data_loader.load_batch(config.training_config.batch_size)
        trajectories = trajectories.to(device)
        conditions = conditions.to(device)
        
        # Training step
        loss = trainer.train_step(trajectories, conditions, optimizer)
        losses.append(loss)
        
        # Logging
        if (epoch + 1) % config.training_config.log_interval == 0:
            avg_loss = sum(losses[-config.training_config.log_interval:]) / config.training_config.log_interval
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Save model
        if (epoch + 1) % config.training_config.save_interval == 0:
            save_path = os.path.join(f"{save_dir}/models", method.get_model_save_name(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'config': config,
                'method': method_name
            }, save_path)
    
    print(f"{method_name.capitalize()} training complete! Models saved to {save_dir}/models")
    return model, trainer, losses


def evaluate_method(method_class, method_name, config, device, model_epoch=1000):
    """
    Evaluate a trained method by generating test trajectories.
    
    Args:
        method_class: Class implementing the diffusion method
        method_name: Name of the method
        config: Experiment configuration
        device: Device to evaluate on
        model_epoch: Epoch of model to load
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {method_name.upper()} DIFFUSION")
    print(f"{'='*60}")
    
    # Load test data
    test_loader = TrajectoryDataLoader(
        config.test_data_dir,
        config.state_dim,
        config.control_dim
    )
    
    # Load trained model
    method = method_class()
    model_path = f"{config.get_save_dir(method_name)}/models/{method.get_model_save_name(model_epoch)}"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Skipping evaluation.")
        return
    
    # Load model using method-specific loader
    if method_name == "vanilla":
        from src.models.vanilla_diffusion import load_vanilla_model
        model, trainer = load_vanilla_model(
            model_path=model_path,
            input_dim=config.model_config.input_dim,
            condition_dim=config.model_config.condition_dim,
            seq_len=config.model_config.seq_len,
            device=device
        )
    elif method_name == "model_based":
        from src.models.model_based_diffusion import load_model_based_model
        model, trainer = load_model_based_model(
            model_path=model_path,
            input_dim=config.model_config.input_dim,
            condition_dim=config.model_config.condition_dim,
            seq_len=config.model_config.seq_len,
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            device=device
        )
    elif method_name == "hankel":
        from src.models.hankel_diffusion import load_hankel_model
        model, trainer = load_hankel_model(
            model_path=model_path,
            input_dim=config.model_config.input_dim,
            condition_dim=config.model_config.condition_dim,
            seq_len=config.model_config.seq_len,
            hankel_data_dir=config.hankel_data_dir,
            state_dim=config.state_dim,
            control_dim=config.control_dim,
            device=device
        )
    
    # Load test trajectories
    import numpy as np
    test_indices = np.arange(0, 20)  # Test on first 20 trajectories
    test_trajectories, test_conditions, test_files = test_loader.load_specific(
        file_indices=test_indices
    )
    
    # Move to device
    test_trajectories = test_trajectories.to(device)
    test_conditions = test_conditions.to(device)
    
    print(f"Generating {len(test_files)} trajectories...")
    
    # Generate samples
    with torch.no_grad():
        seq_len = test_trajectories.shape[1]
        feature_dim = test_trajectories.shape[2]
        generated_samples = trainer.sample((seq_len, feature_dim), test_conditions)
    
    # Save results
    save_dir = f"{config.get_save_dir(method_name)}/trajectories"
    os.makedirs(save_dir, exist_ok=True)
    
    gen_np = generated_samples.cpu().numpy()
    real_np = test_trajectories.cpu().numpy()
    cond_np = test_conditions.cpu().numpy()
    
    for i, original_file in enumerate(test_files):
        file_base = os.path.splitext(original_file)[0]
        
        gen_states = gen_np[i, :, :config.state_dim]
        gen_controls = gen_np[i, :, config.state_dim:]
        real_states = real_np[i, :, :config.state_dim]
        real_controls = real_np[i, :, config.state_dim:]
        
        initial_state = cond_np[i, :config.state_dim]
        target_state = cond_np[i, config.state_dim:2*config.state_dim]
        
        save_path = os.path.join(save_dir, f"{method_name}_diffusion_{file_base}.npz")
        np.savez(
            save_path,
            gen_states=gen_states,
            gen_controls=gen_controls,
            real_states=real_states,
            real_controls=real_controls,
            initial_state=initial_state,
            target_state=target_state,
            original_file=original_file
        )
    
    print(f"Evaluation complete! Results saved to {save_dir}")


def main():
    """Main function to train and evaluate all methods."""
    # Setup
    config = LQRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define methods to train
    methods = [
        (VanillaDiffusion, "vanilla"),
        (ModelBasedDiffusion, "model_based"),
        (HankelDiffusion, "hankel")
    ]
    
    # Training phase
    print("TRAINING PHASE")
    print("="*60)
    
    trained_models = {}
    for method_class, method_name in methods:
        try:
            model, trainer, losses = train_method(
                method_class, method_name, config, device, epochs=1000
            )
            trained_models[method_name] = (model, trainer, losses)
        except Exception as e:
            print(f"Error training {method_name}: {e}")
            continue
    
    # Evaluation phase
    print("\n\nEVALUATION PHASE")
    print("="*60)
    
    for method_class, method_name in methods:
        try:
            evaluate_method(method_class, method_name, config, device, model_epoch=1000)
        except Exception as e:
            print(f"Error evaluating {method_name}: {e}")
            continue
    
    print("\n" + "="*60)
    print("ALL METHODS TRAINING AND EVALUATION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {config.save_dir}/lqr/")


if __name__ == "__main__":
    main()