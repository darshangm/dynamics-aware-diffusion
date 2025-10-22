"""
Evaluation script for trained Diffuser models
"""

import sys
import json
from typing import Optional, Tuple, Any
from datetime import datetime
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import gymnasium_robotics
import gymnasium as gym
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo

from m_diffuser.models.temporal_unet import TemporalUnet
from m_diffuser.models.diffusion import GaussianDiffusion
from m_diffuser.datasets.sequence import SequenceDataset
from m_diffuser.guides.policies import GuidedPolicy, MPCPolicy, DynamicsAwarePolicy
from m_diffuser.utils.training import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Diffuser model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--env', type=str, default='HalfCheetah-v5',
                       help='Gymnasium environment name')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--policy-type', type=str, default='mpc',
                       choices=['guided', 'mpc','dynamics-aware'],
                       help='Policy type to use')
    parser.add_argument('--action-horizon', type=int, default=16,
                       help='Action horizon for MPC policy')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')
    parser.add_argument('--render', type=str, default='none',
                    choices=['none', 'human', 'video'],
                    help='Rendering mode: none (no rendering), human (live display), video (save to file)')
    parser.add_argument('--video-dir', type=str, default='./videos',
                    help='Directory to save videos (when --render video)')
    parser.add_argument('--results-dir', type=str, default='./results',
                    help='Directory to save evaluation results')
    parser.add_argument('--dataset', type=str, default=None,
                   help='Dataset name (if different from checkpoint config)')
    parser.add_argument('--seed', type=int, default=42,
                   help='Random seed')
    parser.add_argument('--sampling-timesteps', type=int, default=200,
                    help='Number of diffusion sampling steps (fewer = faster)')
    
    return parser.parse_args()



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
        # Fallback, though 'betas' should always be there
        n_timesteps = checkpoint.get('config', {}).get('n_timesteps', 200)
    
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
    elif num_levels == 0: # Fallback
        dim_mults = (1, 2, 4, 8)
    else:
        dim_mults = tuple([2**i for i in range(num_levels)])
    
    # Infer base dim from first conv layer
    dim = 128 # Default
    for key in state_dict.keys():
        if 'model.downs.0.0.blocks.0.block.0.weight' in key:
            dim = state_dict[key].shape[0]
            break
    
    # Get other configs from checkpoint if available
    saved_config = checkpoint.get('config', {})
    beta_schedule = saved_config.get('beta_schedule', 'cosine')
    # Horizon must be loaded from the config saved *inside* the checkpoint
    horizon = saved_config.get('horizon', 16) 
    
    config = {
        'dim': dim,
        'dim_mults': list(dim_mults),
        'n_timesteps': n_timesteps,
        'beta_schedule': beta_schedule,
        'horizon': horizon,
    }
    
    return config


def load_model(checkpoint_path: str, dataset_name: str, device: str):
    """
    Load trained model from checkpoint.
    FIXED VERSION: Handles checkpoints without dataset_config/model_config keys.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset_name: Name of dataset for loading normalizer
        device: Device to load model on
    
    Returns:
        (diffusion_model, normalizer)
    """
    # Load checkpoint
    # print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check what keys are in the checkpoint
    # print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    # Try to get config from checkpoint (may not exist in older checkpoints)
    if 'config' in checkpoint:
        print("✓ Found 'config' in checkpoint")
        saved_config = checkpoint['config']
    else:
        print("⚠️  No 'config' key in checkpoint, will infer from weights")
        saved_config = {}
    
    # Infer model architecture from checkpoint weights
    inferred_config = infer_model_config_from_checkpoint(checkpoint)
    
    # print(f"\n✓ Inferred model config:")
    # print(f"  dim: {inferred_config['dim']}")
    # print(f"  dim_mults: {tuple(inferred_config['dim_mults'])}")
    # print(f"  n_timesteps: {inferred_config['n_timesteps']}")
    # print(f"  beta_schedule: {inferred_config['beta_schedule']}")
    # print(f"  horizon: {inferred_config['horizon']}")
    
    # Load dataset for normalizer AND to get dimensions
    # print(f"\nLoading dataset: {dataset_name}")
    dataset = SequenceDataset(
        dataset_name=dataset_name,
        horizon=inferred_config['horizon'],
        normalizer='LimitsNormalizer',
        max_path_length=1000,
        use_padding=True
    )
    
    # print(f"✓ Dataset loaded:")
    # print(f"  observation_dim: {dataset.observation_dim}")
    # print(f"  action_dim: {dataset.action_dim}")
    # print(f"  transition_dim: {dataset.transition_dim}")
    
    # Create model using inferred config + dataset dimensions
    # print("\nCreating model...")
    
    unet = TemporalUnet(
        transition_dim=dataset.transition_dim,
        dim=inferred_config['dim'],
        dim_mults=tuple(inferred_config['dim_mults']),
    )
    
    diffusion = GaussianDiffusion(
        model=unet,
        horizon=inferred_config['horizon'],
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=inferred_config['n_timesteps'],
        beta_schedule=inferred_config['beta_schedule']
    ).to(device)
    
    # Load weights
    print("Loading model weights...")
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.eval()
    
    print("✓ Model loaded successfully!\n")
    
    return diffusion, dataset.normalizer


def evaluate_policy(policy, env, n_episodes: int, render: bool = False):
    """
    Evaluate policy in environment.
    
    Args:
        policy: Policy to evaluate
        env: Gymnasium environment
        n_episodes: Number of episodes to run
        render: Whether to render environment
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        # DEBUG: Print start position and goal
        if isinstance(obs, dict):
            start_pos = obs['observation'][:2]
            goal_pos = obs['desired_goal']
            print(f"\nEpisode {episode + 1}: start={start_pos}, goal={goal_pos}")
            print(f"  Initial distance to goal: {np.linalg.norm(start_pos - goal_pos):.3f}")
        
        while not done and episode_length < 1000:  # Max 1000 steps
            # Get action from policy
            action = policy.get_action(obs)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                env.render()
        
        # DEBUG: Final stats
        if isinstance(obs, dict):
            final_pos = obs['observation'][:2]
            final_distance = np.linalg.norm(final_pos - goal_pos)
            print(f"Episode ended: final_distance={final_distance:.3f}, total_reward={episode_reward}")
            print(f"Distance traveled: start->end = {np.linalg.norm(final_pos - start_pos):.3f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }
    
    return metrics

def save_results(metrics, args, save_dir=None):
    """Save evaluation results to JSON file."""
    # Use provided save_dir or fall back to args.results_dir
    if save_dir is None:
        save_dir = args.results_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create unique filename with timestamp and policy type
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    env_name = args.env.replace('/', '_').replace('-', '_')
    filename = f"{args.policy_type}_{env_name}_{timestamp}.json"
    filepath = os.path.join(save_dir, filename)
    
    # Prepare results dict
    results = {
        'policy_type': args.policy_type,
        'environment': args.env,
        'checkpoint': args.checkpoint,
        'dataset': args.dataset if hasattr(args, 'dataset') else None,
        'n_episodes': args.n_episodes,
        'sampling_timesteps': args.sampling_timesteps,
        'seed': args.seed,
        'timestamp': timestamp,
        'metrics': {
            'mean_reward': float(metrics['mean_reward']),
            'std_reward': float(metrics['std_reward']),
            'mean_length': float(metrics['mean_length']),
            'std_length': float(metrics['std_length']),
            'episode_rewards': [float(r) for r in metrics['episode_rewards']],
            'episode_lengths': [int(l) for l in metrics['episode_lengths']],
        }
    }
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {filepath}")
    return filepath


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Evaluating Diffuser Model")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env}")
    print(f"Policy type: {args.policy_type}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Infer dataset name from environment
    # Map Gymnasium env names to Minari dataset names
    env_to_dataset = {
        'HalfCheetah-v5': 'halfcheetah-medium-v0',
        'Hopper-v5': 'hopper-medium-v0',
        'Walker2d-v5': 'walker2d-medium-v0',
    }
    
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = env_to_dataset.get(args.env)
        if dataset_name is None:
            print(f"Warning: No default dataset for {args.env}, using mujoco/halfcheetah/simple-v0")
            dataset_name = 'mujoco/halfcheetah/simple-v0'
    
    # Load model
    diffusion, normalizer = load_model(args.checkpoint, dataset_name, args.device)

    # Override sampling timesteps for faster inference
    original_timesteps = diffusion.n_timesteps
    diffusion.n_timesteps = args.sampling_timesteps
    print(f"✓ Using {args.sampling_timesteps} sampling timesteps for inference (trained with {original_timesteps})")
    
    # Create policy
    if args.policy_type == 'guided':
        policy = GuidedPolicy(diffusion, normalizer)
        print("✓ Created GuidedPolicy")
    elif args.policy_type == 'mpc':
        policy = MPCPolicy(diffusion, normalizer, action_horizon=args.action_horizon)
        print(f"✓ Created MPCPolicy (action_horizon={args.action_horizon})")
    elif args.policy_type == 'dynamics-aware':
        print("\n" + "="*60)
        print("Creating DYNAMICS-AWARE policy")
        print("="*60)
        
        # Extract dynamics matrices
        from m_diffuser.dynamics.registry import get_dynamics_for_env
        from m_diffuser.dynamics.projection import ProjectionMatrixBuilder
        
        print(f"Extracting dynamics for {args.env}...")
        A, B, state_dim, action_dim = get_dynamics_for_env(
            env_name=args.env,
            dataset_name=dataset_name,
            method='data_driven'
        )
        
        print(f"✓ Dynamics extracted:")
        print(f"  A shape: {A.shape}")
        print(f"  B shape: {B.shape}")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        
        # Build projection matrix
        horizon = diffusion.horizon
        print(f"\nBuilding projection matrix (horizon={horizon})...")
        builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
        P = builder.get_projection_matrix(horizon)
        
        # Create dynamics-aware policy
        policy = DynamicsAwarePolicy(
            diffusion_model=diffusion,
            projection_matrix=P,
            normalizer=normalizer,
            state_dim=state_dim,
            observation_dim=diffusion.observation_dim,
            action_dim=diffusion.action_dim,
            horizon=horizon,
            projection_schedule='noise_schedule',
            projection_strength=1.0,
            action_horizon=args.action_horizon
        )
        
        print("✓ DynamicsAwarePolicy created successfully!")
        print("="*60)
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")
    
    # Create environment
    if args.render == 'human':
        env = gym.make(args.env, render_mode='human')
    elif args.render == 'video':
        from gymnasium.wrappers import RecordVideo
        env = gym.make(args.env, render_mode='rgb_array')
        env = RecordVideo(env, args.video_dir, episode_trigger=lambda x: True)
        print(f"Recording videos to: {args.video_dir}")
    else:  # args.render == 'none'
        env = gym.make(args.env)

    env.reset(seed=args.seed)
    print(f"✓ Created environment: {args.env}")
    
    # Evaluate
    print(f"\n{'='*60}")
    print(f"Running {args.n_episodes} evaluation episodes...")
    print(f"{'='*60}\n")
    metrics = evaluate_policy(policy, env, args.n_episodes)

    results_file = save_results(metrics, args)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Mean Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    print("=" * 60)
    
    env.close()




if __name__ == '__main__':
    main()