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
    """
    Main trainer class for diffusion models.
    Handles training loop, logging, and checkpointing.
    """
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 scheduler: Optional[_LRScheduler] = None,
                 device: str = 'cuda',
                 log_dir: str = './logs',
                 save_freq: int = 10000,
                 eval_freq: int = 5000,
                 use_ema: bool = True,
                 ema_decay: float = 0.995,
                 gradient_clip: Optional[float] = 1.0):
        """
        Args:
            model: Diffusion model to train
            optimizer: Optimizer
            train_loader: Training data loader
            val_loader: Optional validation data loader
            scheduler: Optional learning rate scheduler
            device: Device to train on
            log_dir: Directory for logs and checkpoints
            save_freq: Steps between checkpoint saves
            eval_freq: Steps between evaluations
            use_ema: Whether to use EMA
            ema_decay: EMA decay rate
            gradient_clip: Gradient clipping value (None to disable)
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.log_dir = Path(log_dir)
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.gradient_clip = gradient_clip
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # EMA
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        # Metrics tracking
        self.metrics = defaultdict(list)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step.
        
        Args:
            batch: Batch of data
        
        Returns:
            Loss value
        """
        self.model.train()
        
        # Move batch to device
        conditions = batch['conditions'].to(self.device)
        
        # Forward pass
        loss = self.model(conditions)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        
        self.optimizer.step()
        
        # Update EMA
        if self.ema is not None:
            self.ema.update()
        
        return loss.item()
    
    @torch.no_grad()
    def eval_step(self) -> float:
        """
        Evaluation step.
        
        Returns:
            Average validation loss
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        
        # Apply EMA if available
        if self.ema is not None:
            self.ema.apply_shadow()
        
        losses = []
        for batch in self.val_loader:
            conditions = batch['conditions'].to(self.device)
            loss = self.model(conditions)
            losses.append(loss.item())
        
        # Restore original parameters
        if self.ema is not None:
            self.ema.restore()
        
        return np.mean(losses)
    
    def train(self, n_epochs: int):
        """
        Main training loop.
        
        Args:
            n_epochs: Number of epochs to train
        """
        print(f"Starting training for {n_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Log directory: {self.log_dir}")
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.step += 1
                
                # Logging
                if self.step % 100 == 0:
                    avg_loss = np.mean(epoch_losses[-100:])
                    print(f"Epoch {epoch} | Step {self.step} | Loss: {avg_loss:.4f}")
                    self.metrics['train_loss'].append(avg_loss)
                
                # Evaluation
                if self.step % self.eval_freq == 0:
                    val_loss = self.eval_step()
                    print(f"Step {self.step} | Val Loss: {val_loss:.4f}")
                    self.metrics['val_loss'].append(val_loss)
                
                # Save checkpoint
                if self.step % self.save_freq == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.step}.pt')
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # End of epoch summary
            avg_epoch_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")
            self.train_losses.append(avg_epoch_loss)
        
        # Final save
        self.save_checkpoint('final_model.pt')
        print("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """
        Save model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.log_dir / filename
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'metrics': dict(self.metrics),
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.metrics = defaultdict(list, checkpoint['metrics'])
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from step {self.step}, epoch {self.epoch}")


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