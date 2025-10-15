"""
Training script for Diffuser model.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from m_diffuser.models.temporal_unet import TemporalUnet
from m_diffuser.models.diffusion import GaussianDiffusion
from m_diffuser.datasets.sequence import SequenceDataset, create_dataloader
from m_diffuser.utils.training import Trainer, CosineAnnealingWarmup, count_parameters, save_config
from m_diffuser.utils.arrays import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Diffuser model')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v0',
                       help='Minari dataset name')
    parser.add_argument('--horizon', type=int, default=8,
                       help='Planning horizon length')
    
    # Model
    parser.add_argument('--dim', type=int, default=128,
                       help='Base model dimension')
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Dimension multipliers for U-Net levels')
    parser.add_argument('--n-timesteps', type=int, default=500,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta-schedule', type=str, default='cosine',
                       choices=['linear', 'cosine'],
                       help='Noise schedule')
    
    # Training
    parser.add_argument('--n-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=2000,
                       help='Warmup steps for learning rate')
    parser.add_argument('--gradient-clip', type=float, default=3.0,
                       help='Gradient clipping value')
    
    # EMA
    parser.add_argument('--use-ema', action='store_true', default=True,
                       help='Use exponential moving average')
    parser.add_argument('--ema-decay', type=float, default=0.995,
                       help='EMA decay rate')
    
    # Logging
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory for logs and checkpoints')
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Steps between checkpoint saves')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Steps between evaluations')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Training Diffuser Model")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Horizon: {args.horizon}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    # Create log directory
    log_dir = Path(args.log_dir) / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = log_dir / 'config.json'
    save_config(vars(args), str(config_path))
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = SequenceDataset(
        dataset_name=args.dataset,
        horizon=args.horizon,
    )
    
    # Create data loader
    train_loader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Observation dim: {dataset.observation_dim}")
    print(f"Action dim: {dataset.action_dim}")
    print(f"Transition dim: {dataset.transition_dim}")
    
    # Create model
    print("\nCreating model...")
    unet = TemporalUnet(
        transition_dim=dataset.transition_dim,
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        horizon=args.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=args.n_timesteps,
        beta_schedule=args.beta_schedule,
    )
    
    n_params = count_parameters(diffusion)
    print(f"Model parameters: {n_params:,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.lr)
    
    # Create scheduler
    total_steps = args.n_epochs * len(train_loader)
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
    )
    
    # Create trainer
    trainer = Trainer(
        model=diffusion,
        optimizer=optimizer,
        train_loader=train_loader,
        scheduler=scheduler,
        device=args.device,
        log_dir=str(log_dir),
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        gradient_clip=args.gradient_clip,
    )
    
    # Train
    print("\nStarting training...\n")
    trainer.train(args.n_epochs)
    
    print("\nTraining complete!")
    print(f"Logs and checkpoints saved to: {log_dir}")


if __name__ == '__main__':
    main()