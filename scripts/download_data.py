"""
Download and inspect Minari datasets.
"""

import minari
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download Minari datasets')
    
    parser.add_argument('--dataset', type=str, default=None,
                       help='Specific dataset to download (e.g., halfcheetah-medium-v0)')
    parser.add_argument('--list', action='store_true',
                       help='List all available datasets')
    parser.add_argument('--info', type=str, default=None,
                       help='Show info for specific dataset')
    
    return parser.parse_args()


def list_datasets():
    """List all available Minari datasets."""
    print("=" * 60)
    print("Available Minari Datasets")
    print("=" * 60)
    
    datasets = minari.list_remote_datasets()
    
    # Group by environment
    env_groups = {}
    for dataset in datasets:
        env_name = dataset.split('-')[0]
        if env_name not in env_groups:
            env_groups[env_name] = []
        env_groups[env_name].append(dataset)
    
    for env_name, dataset_list in sorted(env_groups.items()):
        print(f"\n{env_name.upper()}:")
        for dataset in sorted(dataset_list):
            print(f"  - {dataset}")
    
    print("\n" + "=" * 60)
    print(f"Total datasets: {len(datasets)}")
    print("=" * 60)


def show_dataset_info(dataset_name: str):
    """Show information about a specific dataset."""
    print("=" * 60)
    print(f"Dataset Info: {dataset_name}")
    print("=" * 60)
    
    try:
        # Try to load dataset
        dataset = minari.load_dataset(dataset_name)
        
        print(f"\nDataset ID: {dataset.spec.id}")
        print(f"Total episodes: {dataset.total_episodes}")
        print(f"Total steps: {dataset.total_steps}")
        
        # Get environment info
        print(f"\nEnvironment: {dataset.spec.env_spec.id if hasattr(dataset.spec, 'env_spec') else 'N/A'}")
        
        # Sample first episode
        if dataset.total_episodes > 0:
            episode = next(iter(dataset))
            print(f"\nSample Episode:")
            print(f"  - Observations shape: {episode.observations.shape}")
            print(f"  - Actions shape: {episode.actions.shape}")
            print(f"  - Rewards shape: {episode.rewards.shape}")
            print(f"  - Episode length: {len(episode.rewards)}")
            print(f"  - Total reward: {episode.rewards.sum():.2f}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying to download dataset...")
        try:
            dataset = minari.download_dataset(dataset_name)
            print(f"✓ Dataset downloaded successfully!")
            show_dataset_info(dataset_name)
        except Exception as e2:
            print(f"Error downloading dataset: {e2}")


def download_dataset(dataset_name: str):
    """Download a specific dataset."""
    print(f"Downloading dataset: {dataset_name}")
    print("=" * 60)
    
    try:
        # Try to load first (in case already downloaded)
        dataset = minari.load_dataset(dataset_name)
        print(f"\n✓ Dataset already exists locally!")
        print(f"Dataset location: {dataset.spec.data_path}")
        print(f"\nTotal episodes: {dataset.total_episodes}")
        print(f"Total steps: {dataset.total_steps}")
        
    except Exception as e:
        # If not found, download it
        print(f"Dataset not found locally. Downloading...")
        try:
            minari.download_dataset(dataset_name)
            # Load after download
            dataset = minari.load_dataset(dataset_name)
            print(f"\n✓ Dataset downloaded successfully!")
            print(f"Dataset location: {dataset.spec.data_path}")
            print(f"\nTotal episodes: {dataset.total_episodes}")
            print(f"Total steps: {dataset.total_steps}")
        except Exception as e2:
            print(f"Error downloading dataset: {e2}")


def download_common_datasets():
    """Download commonly used datasets for locomotion tasks."""
    common_datasets = [
        'halfcheetah-medium-v0',
        'hopper-medium-v0',
        'walker2d-medium-v0',
    ]
    
    print("Downloading common locomotion datasets...")
    print("=" * 60)
    
    for dataset_name in common_datasets:
        try:
            print(f"\n{dataset_name}:")
            # Check if already downloaded
            try:
                dataset = minari.load_dataset(dataset_name)
                print(f"  ✓ Already downloaded ({dataset.total_episodes} episodes)")
            except:
                print(f"  Downloading...")
                dataset = minari.download_dataset(dataset_name)
                print(f"  ✓ Downloaded ({dataset.total_episodes} episodes)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Done!")


def main():
    """Main function."""
    args = parse_args()
    
    if args.list:
        list_datasets()
    elif args.info:
        show_dataset_info(args.info)
    elif args.dataset:
        download_dataset(args.dataset)
    else:
        # Default: download common datasets
        print("No specific action specified. Downloading common datasets...\n")
        download_common_datasets()


if __name__ == '__main__':
    main()