"""
Evaluation script for trained Diffuser models.
"""

import sys
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
    parser.add_argument('--policy-type', type=str, default='guided',
                       choices=['guided', 'mpc','dynamics-aware'],
                       help='Policy type to use')
    parser.add_argument('--action-horizon', type=int, default=8,
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
    parser.add_argument('--sampling-timesteps', type=int, default=100,
                    help='Number of diffusion sampling steps (fewer = faster)')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, dataset_name: str, device: str):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset_name: Name of dataset for loading normalizer
        device: Device to load model on
    
    Returns:
        (diffusion_model, normalizer)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config (if available)
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / 'config.json'
    
    if config_path.exists():
        config = load_config(str(config_path))
        print("Loaded config from checkpoint directory")
    else:
        # Use default config
        print("Warning: config.json not found, using defaults")
        config = {
            'dim': 128,
            'dim_mults': [1, 2, 4, 8],
            'n_timesteps': 1000,
            'beta_schedule': 'cosine',
            'horizon': 64,
        }
    
    # Load dataset for normalizer
    print(f"Loading dataset: {dataset_name}")
    dataset = SequenceDataset(
        dataset_name=dataset_name,
        horizon=config['horizon'],
    )
    
    # Create model
    print("Creating model...")
    unet = TemporalUnet(
        transition_dim=dataset.transition_dim,
        dim=config['dim'],
        dim_mults=tuple(config['dim_mults']),
    )

    
    diffusion = GaussianDiffusion(
        model=unet,
        horizon=config['horizon'],
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=config['n_timesteps'],
        beta_schedule=config['beta_schedule'],
    )

    # Load weights
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion.to(device)
    diffusion.eval()
    
    print("Model loaded successfully!")
    
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
    
    for episode in tqdm(range(n_episodes), desc='Evaluating'):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        
        # DEBUG: Print initial state
        print(f"\n=== Episode {episode + 1} ===")
        if isinstance(obs, dict):
            print(f"Start position: {obs['observation'][:2]}")
            print(f"Goal position: {obs['desired_goal']}")
            start_pos = obs['observation'][:2].copy()
            goal_pos = obs['desired_goal'].copy()
            initial_distance = np.linalg.norm(start_pos - goal_pos)
            print(f"Initial distance to goal: {initial_distance:.3f}")
        else:
            start_pos = None
            goal_pos = None
        
        while not (done or truncated):
            # Get action from policy
            action = policy.get_action(obs)
            
            # DEBUG: Print first few actions
            if episode_length < 3:
                print(f"Step {episode_length}: action={action}, magnitude={np.linalg.norm(action):.4f}")
            
            # Clip action to environment bounds
            if hasattr(env.action_space, 'low'):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # DEBUG: Print when reward received or first few steps
            # if (reward > 0 or episode_length < 3) and isinstance(obs, dict):
            #     curr_pos = obs['observation'][:2]
            #     curr_distance = np.linalg.norm(curr_pos - goal_pos)
            #     print(f"  After step: pos={curr_pos}, distance={curr_distance:.3f}, reward={reward}")
            
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
    diffusion.n_timesteps = args.sampling_timesteps
    print(f"Using {args.sampling_timesteps} sampling timesteps for inference")
    
    # Create policy
    if args.policy_type == 'guided':
        policy = GuidedPolicy(diffusion, normalizer)
    elif args.policy_type == 'mpc':
        policy = MPCPolicy(diffusion, normalizer, action_horizon=args.action_horizon)
    elif args.policy_type == 'dynamics-aware':
        # Extract dynamics for environment
        from m_diffuser.dynamics import get_dynamics_for_env, ProjectionMatrixBuilder
        
        print("Extracting dynamics for environment...")
        A, B, state_dim, action_dim_dynamics = get_dynamics_for_env(args.env)
        
        print(f"  State dim: {state_dim}, Action dim: {action_dim_dynamics}")
        print(f"  A matrix shape: {A.shape}, B matrix shape: {B.shape}")
        
        print(f"Building projection matrix for horizon={diffusion.horizon}...")
        proj_builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim_dynamics)
        P = proj_builder.get_projection_matrix(diffusion.horizon)
        
        print(f"  Projection matrix P shape: {P.shape}")
        print(f"  P is projection: {torch.allclose(P @ P, P, atol=1e-4)}")
        
        policy = DynamicsAwarePolicy(
            diffusion_model=diffusion,
            normalizer=normalizer,
            projection_matrix=P,
            state_dim=state_dim,
            action_dim=action_dim_dynamics
        )
    else:
        raise ValueError(f"Unknown policy type: {args.policy_type}")
    
    print(f"\nCreated {args.policy_type} policy")
    
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

    env.reset(seed=42)



    print(f"Created environment: {args.env}")
    
    # Evaluate
    print(f"\nRunning {args.n_episodes} evaluation episodes...\n")
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