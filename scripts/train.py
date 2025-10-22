"""
Unified training, supports training from scratch and fine-tuning with various loss objectives.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import json

from m_diffuser.models.temporal_unet import TemporalUnet
from m_diffuser.models.diffusion import GaussianDiffusion
from m_diffuser.datasets.sequence import SequenceDataset, create_dataloader
from m_diffuser.utils.training import Trainer, CosineAnnealingWarmup, count_parameters, save_config
from m_diffuser.utils.arrays import set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train/Fine-tune Diffuser model')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-v0',
                       help='Minari dataset name')
    parser.add_argument('--horizon', type=int, default=16,
                       help='Planning horizon length')
    
    # Model
    parser.add_argument('--dim', type=int, default=128,
                       help='Base model dimension')
    parser.add_argument('--dim-mults', type=int, nargs='+', default=[1, 2, 4],
                       help='Dimension multipliers for U-Net levels')
    parser.add_argument('--n-timesteps', type=int, default=200,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta-schedule', type=str, default='cosine',
                       choices=['linear', 'cosine'],
                       help='Noise schedule')
    
    # Training
    parser.add_argument('--n-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=2000,
                       help='Warmup steps for learning rate')
    parser.add_argument('--gradient-clip', type=float, default=4.0,
                       help='Gradient clipping value')
    
    # Fine-tuning
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to resume from or fine-tune')
    parser.add_argument('--reset-optimizer', action='store_true',
                       help='Reset optimizer state when loading checkpoint (for fine-tuning)')
    parser.add_argument('--finetune-mode', action='store_true',
                       help='Enable fine-tuning mode (lower LR, reset scheduler)')
    
    # Loss composition
    parser.add_argument('--projection-weight', type=float, default=0.0,
                       help='Weight for dynamics projection loss (0 = disabled)')
    parser.add_argument('--value-guidance-weight', type=float, default=0.0,
                       help='Weight for value guidance loss (0 = disabled)')
    # Add more loss types here as you develop them
    
    # Dynamics extraction (for projection loss)
    parser.add_argument('--env', type=str, default='PointMaze_UMaze-v3',
                       help='Environment name (for dynamics extraction)')
    parser.add_argument('--dynamics-method', type=str, default='data-driven',
                       choices=['data-driven', 'analytical', 'none'],
                       help='Method for dynamics extraction')
    
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
    parser.add_argument('--run-name', type=str, default=None,
                       help='Name for this run (for organizing logs)')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def load_config_from_checkpoint(checkpoint_path):
    """Load config from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def infer_model_config_from_checkpoint(checkpoint):
    """
    Infer model configuration from checkpoint weights.
    More reliable than trusting saved config.
    """
    state_dict = checkpoint['model_state_dict']
    
    # Infer n_timesteps from beta shape
    if 'betas' in state_dict:
        n_timesteps = state_dict['betas'].shape[0]
    else:
        n_timesteps = 500
    
    # Infer dim_mults by counting encoder levels
    max_down_idx = -1
    for key in state_dict.keys():
        if 'model.downs.' in key:
            parts = key.split('.')
            if len(parts) > 2 and parts[2].isdigit():
                idx = int(parts[2])
                max_down_idx = max(max_down_idx, idx)
    
    num_levels = max_down_idx + 1
    
    # Default dim_mults for different number of levels
    if num_levels == 3:
        dim_mults = (1, 2, 4)
    elif num_levels == 4:
        dim_mults = (1, 2, 4, 8)
    elif num_levels == 2:
        dim_mults = (1, 2)
    else:
        dim_mults = tuple([2**i for i in range(num_levels)])
    
    # Infer base dim from first conv layer
    for key in state_dict.keys():
        if 'model.downs.0.0.blocks.0.block.0.weight' in key:
            dim = state_dict[key].shape[0]
            break
    else:
        dim = 128
    
    # Get other configs from checkpoint if available
    saved_config = checkpoint.get('config', {})
    beta_schedule = saved_config.get('beta_schedule', 'cosine')
    horizon = saved_config.get('horizon', 16)
    
    config = {
        'dim': dim,
        'dim_mults': list(dim_mults),
        'n_timesteps': n_timesteps,
        'beta_schedule': beta_schedule,
        'horizon': horizon,
    }
    
    return config


def create_model(args, dataset, checkpoint=None):
    """
    Create model, optionally loading from checkpoint.
    
    Returns:
        (diffusion_model, config_dict)
    """
    if checkpoint is not None:
        print("Loading model architecture from checkpoint...")
        
        # Infer config from actual weights (most reliable!)
        config = infer_model_config_from_checkpoint(checkpoint)
        
        dim = config['dim']
        dim_mults = tuple(config['dim_mults'])
        n_timesteps = config['n_timesteps']
        beta_schedule = config['beta_schedule']
        horizon = config['horizon']
        
        print(f"Inferred checkpoint config:")
        print(f"  dim={dim}, dim_mults={dim_mults}")
        print(f"  n_timesteps={n_timesteps}, beta_schedule={beta_schedule}")
        print(f"  horizon={horizon}")
        
    else:
        # Use command line args for new model
        dim = args.dim
        dim_mults = tuple(args.dim_mults)
        n_timesteps = args.n_timesteps
        beta_schedule = args.beta_schedule
        horizon = args.horizon
        
        config = {
            'dim': dim,
            'dim_mults': list(dim_mults),
            'n_timesteps': n_timesteps,
            'beta_schedule': beta_schedule,
            'horizon': horizon,
        }
    
    # Create model
    unet = TemporalUnet(
        transition_dim=dataset.transition_dim,
        dim=dim,
        dim_mults=dim_mults,
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        horizon=horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=n_timesteps,
        beta_schedule=beta_schedule,
    )
    
    # Load weights if checkpoint provided
    if checkpoint is not None:
        print("Loading model weights from checkpoint...")
        diffusion.load_state_dict(checkpoint['model_state_dict'])
    
    return diffusion, config

def build_loss_function(args, diffusion, dataset):
    """
    Build composed loss function based on arguments.
    
    Returns:
        loss_fn: Function that takes a batch and returns (total_loss, loss_dict)
    """
    from m_diffuser.losses import (
        DiffusionLoss,
        ProjectionLoss,
        ComposedLoss
    )
    
    # Base diffusion loss (always included)
    losses = [DiffusionLoss(diffusion)]
    loss_names = ['diffusion']
    
    # Add projection loss if requested
    if args.projection_weight > 0:
        print(f"\nAdding projection loss (weight={args.projection_weight})...")
        
        from m_diffuser.dynamics import get_dynamics_for_env
        from m_diffuser.dynamics.projection import ProjectionMatrixBuilder
        
        # Extract dynamics
        print(f"Extracting dynamics using {args.dynamics_method} method...")
        A, B, state_dim, action_dim = get_dynamics_for_env(
            env_name='PointMaze_UMaze-v3',
            dataset_name=args.dataset,  # 'D4RL/pointmaze/umaze-v2'
            method='data_driven'  # Use fitted dynamics!
        )
        
        # Build projection matrix
        print(f"Building projection matrix for horizon={diffusion.horizon}...")
        proj_builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
        P = proj_builder.get_projection_matrix(diffusion.horizon)
        P = P.to(args.device)
        # Create projection loss
        proj_loss = ProjectionLoss(
            projection_matrix=P,
            normalizer=dataset.normalizer,
            state_dim=state_dim,
            action_dim=action_dim,
            observation_dim=dataset.observation_dim,
            horizon=diffusion.horizon,
            weight=args.projection_weight,
            device=args.device
        )
        
        losses.append(proj_loss)
        loss_names.append('projection')
        
        print(f"✓ Projection loss configured")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim_dynamics}")
        print(f"  Observation dim: {dataset.observation_dim}")
    
    # Add value guidance loss if requested
    if args.value_guidance_weight > 0:
        print(f"\nAdding value guidance loss (weight={args.value_guidance_weight})...")
        # TODO: Implement when you add value-based guidance
        print("WARNING: Value guidance not yet implemented")
    
    # TODO: Add more loss types here as you develop them
    # if args.some_other_loss_weight > 0:
    #     losses.append(SomeOtherLoss(...))
    #     loss_names.append('some_other')
    
    # Compose all losses
    if len(losses) == 1:
        # Just diffusion loss
        return losses[0], loss_names
    else:
        # Multiple losses
        return ComposedLoss(losses), loss_names


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Determine mode
    mode = "Fine-tuning" if args.checkpoint else "Training"
    
    print("=" * 60)
    print(f"{mode} Diffuser Model")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Horizon: {args.horizon}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Reset optimizer: {args.reset_optimizer}")
    
    # Loss composition
    if args.projection_weight > 0:
        print(f"Projection loss weight: {args.projection_weight}")
    if args.value_guidance_weight > 0:
        print(f"Value guidance weight: {args.value_guidance_weight}")
    
    print("=" * 60)
    
    # Load checkpoint if provided
    checkpoint = None
    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        print(f"✓ Checkpoint loaded (trained for {checkpoint.get('epoch', '?')} epochs)")
    
    # Create log directory
    if args.run_name:
        log_dir = Path(args.log_dir) / args.dataset / args.run_name
    else:
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
    diffusion, model_config = create_model(args, dataset, checkpoint)
    
    n_params = count_parameters(diffusion)
    print(f"Model parameters: {n_params:,}")
    print(f"Model config: {model_config}")
    
    # Build loss function
    loss_fn, loss_names = build_loss_function(args, diffusion, dataset)
    print(f"\nLoss components: {loss_names}")
    
    # Create optimizer
    lr = args.lr
    if args.finetune_mode and not args.reset_optimizer:
        # Lower learning rate for fine-tuning by default
        lr = args.lr * 0.1
        print(f"Fine-tuning mode: reduced LR to {lr}")
    
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    
    # Load optimizer state if continuing training (not fine-tuning)
    if checkpoint is not None and not args.reset_optimizer:
        print("Loading optimizer state from checkpoint...")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Create scheduler
    total_steps = args.n_epochs * len(train_loader)
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_steps=args.warmup_steps if args.reset_optimizer else 0,
        total_steps=total_steps,
    )
    
    # Load scheduler state if continuing
    if checkpoint is not None and not args.reset_optimizer and 'scheduler_state_dict' in checkpoint:
        print("Loading scheduler state from checkpoint...")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Create trainer with custom loss function
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
        loss_fn=loss_fn,  # Pass custom loss function
        loss_names=loss_names,  # For logging
    )
    
    # Set starting epoch if resuming
    start_epoch = 0
    if checkpoint is not None and not args.reset_optimizer:
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}")
    
    # Train
    print(f"\nStarting {mode.lower()}...\n")
    trainer.train(args.n_epochs, start_epoch=start_epoch)
    
    print(f"\n{mode} complete!")
    print(f"Logs and checkpoints saved to: {log_dir}")
    
    # Save final model config
    final_config = {
        **model_config,
        'projection_weight': args.projection_weight,
        'value_guidance_weight': args.value_guidance_weight,
        'loss_components': loss_names,
    }
    
    final_config_path = log_dir / 'final_config.json'
    with open(final_config_path, 'w') as f:
        json.dump(final_config, f, indent=2)
    print(f"Final config saved to: {final_config_path}")


if __name__ == '__main__':
    main()