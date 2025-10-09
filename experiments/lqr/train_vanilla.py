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