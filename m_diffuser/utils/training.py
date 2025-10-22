"""
Training utilities for diffusion models.
Includes logging, checkpointing, and training loop helpers.
"""

import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Optional, Any
import json
from pathlib import Path
import numpy as np
from collections import defaultdict


class EMA:
    """
    Exponential Moving Average for model parameters.
    Helps stabilize training and improve sample quality.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.995):
        """
        Args:
            model: Model to track
            decay: EMA decay rate (higher = slower update)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """Updated Trainer class with custom loss function support."""
    
    def __init__(self,
                 model,
                 optimizer,
                 train_loader,
                 scheduler=None,
                 device='cuda',
                 log_dir='./logs',
                 save_freq=10000,
                 eval_freq=5000,
                 use_ema=True,
                 ema_decay=0.995,
                 gradient_clip=1.0,
                 loss_fn=None,           # NEW: Custom loss function
                 loss_names=None):       # NEW: Names for logging
        
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.device = device
        self.log_dir = log_dir
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.use_ema = use_ema
        self.gradient_clip = gradient_clip
        
        # NEW: Custom loss function
        self.loss_fn = loss_fn
        self.loss_names = loss_names if loss_names else ['loss']
        
        # Move model to device
        self.model.to(device)
        
        # EMA setup
        if use_ema:
            from copy import deepcopy
            self.ema_model = deepcopy(model)
            self.ema_decay = ema_decay
        else:
            self.ema_model = None
        
        self.global_step = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = open(os.path.join(log_dir, 'training.log'), 'a')
    
    def compute_loss(self, batch):
        """
        Compute loss for a batch.
        
        Returns:
            loss: Scalar tensor
            loss_dict: Dictionary of loss components (for logging)
        """
        if self.loss_fn is None:
            # Default: use model's built-in loss
            loss = self.model.loss(batch['conditions'])
            loss_dict = {'diffusion': loss.item()}
        else:
            # Use custom loss function
            # Check if it returns tuple (composed loss) or scalar
            loss_output = self.loss_fn(batch)
            
            if isinstance(loss_output, tuple):
                # Composed loss returns (total_loss, loss_dict)
                loss, loss_dict = loss_output
            else:
                # Single loss function
                loss = loss_output
                loss_dict = {self.loss_names[0]: loss.item()}
        
        return loss, loss_dict
    
    def train_step(self, batch):
        """Single training step."""
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        
        # Forward pass
        loss, loss_dict = self.compute_loss(batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.gradient_clip
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # EMA update
        if self.ema_model is not None:
            self.update_ema()
        
        self.global_step += 1
        
        return loss_dict
    
    def update_ema(self):
        """Update EMA model."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), 
                self.model.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    param.data, alpha=1 - self.ema_decay
                )
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'horizon': self.model.horizon,
                'observation_dim': self.model.observation_dim,
                'action_dim': self.model.action_dim,
                'n_timesteps': self.model.n_timesteps,
                'beta_schedule': self.model.beta_schedule,
            }
        }
        
        if self.ema_model is not None:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        save_path = os.path.join(
            self.log_dir, 
            f'checkpoint_step_{self.global_step}.pt'
        )
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(self.log_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
        
        return save_path
    
    def train(self, n_epochs, start_epoch=0):
        """Main training loop."""
        from tqdm import tqdm
        
        self.model.train()
        
        for epoch in range(start_epoch, start_epoch + n_epochs):
            epoch_losses = {name: [] for name in self.loss_names}
            
            pbar = tqdm(
                self.train_loader, 
                desc=f'Epoch {epoch+1}/{start_epoch + n_epochs}'
            )
            
            for batch in pbar:
                loss_dict = self.train_step(batch)
                
                # Accumulate losses
                for name in self.loss_names:
                    if name in loss_dict:
                        epoch_losses[name].append(loss_dict[name])
                
                # Update progress bar
                pbar_dict = {
                    name: f"{loss_dict.get(name, 0):.4f}" 
                    for name in self.loss_names
                }
                pbar_dict['lr'] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
                pbar.set_postfix(pbar_dict)
                
                # Save checkpoint
                if self.global_step % self.save_freq == 0:
                    save_path = self.save_checkpoint(epoch)
                    print(f"\n✓ Saved checkpoint: {save_path}")
            
            # Epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            for name in self.loss_names:
                if epoch_losses[name]:
                    avg_loss = sum(epoch_losses[name]) / len(epoch_losses[name])
                    print(f"  {name.capitalize()} loss: {avg_loss:.4f}")
                    
                    # Log to file
                    self.log_file.write(
                        f"Epoch {epoch+1}, {name}: {avg_loss:.4f}\n"
                    )
            
            self.log_file.flush()
        
        # Final checkpoint
        final_path = self.save_checkpoint(start_epoch + n_epochs - 1)
        print(f"\n✓ Final checkpoint saved: {final_path}")
        
        self.log_file.close()




class CosineAnnealingWarmup(_LRScheduler):
    """
    Cosine annealing with linear warmup.
    """
    
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 0.0,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate."""
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_scale = step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return [base_lr * lr_scale + self.min_lr * (1 - lr_scale) for base_lr in self.base_lrs]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved: {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def create_trainer_with_custom_loss(model, optimizer, train_loader, scheduler,
                                    device, log_dir, save_freq, eval_freq,
                                    use_ema, ema_decay, gradient_clip,
                                    loss_fn=None, loss_names=None):
    """
    Function to create trainer with custom loss.
    Use this in train.py script.
    """
    return Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        device=device,
        log_dir=log_dir,
        save_freq=save_freq,
        eval_freq=eval_freq,
        use_ema=use_ema,
        ema_decay=ema_decay,
        gradient_clip=gradient_clip,
        loss_fn=loss_fn,
        loss_names=loss_names
    )
