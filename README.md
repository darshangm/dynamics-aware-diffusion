# Dynamics-aware Diffusion Models for Planning and Control


## Repository based on
 **Planning with Diffusion for Flexible Behavior Synthesis**  
> Michael Janner, Yilun Du, Joshua Tenenbaum, Sergey Levine  
> ICML 2022 
diffusion-planner.github.io

This implementation modernizes the original Diffuser codebase and adds dynamics-aware extensions, with modern Gymnasium and RL datasets:
- **Gymnasium** (instead of deprecated gym)
- **Minari** datasets (instead of D4RL)
- **Modern MuJoCo** (instead of mujoco-py)
- **Dynamics-aware projection** (ensures physical feasibility)
- **Windows compatibility**

---

## Key Feature: Dynamics-Aware Diffusion

Standard diffusion models can generate trajectories that **violate physical constraints**. Our dynamics-aware extension integrates system dynamics directly into the sampling process via **projection-based denoising**:
```
Standard:         x_{i-1} = denoise(x_i)
Dynamics-aware:   x_{i-1} = project(denoise(x_i), dynamics)
```

**Benefits:**
-  Generated trajectories satisfy dynamics constraints
-  No retraining required (plug-and-play at inference)
-  Works with both known and unknown dynamics
-  Improves trajectory feasibility and task performance

---

## Quick Start Workflow

### Step 1: Installation

### Prerequisites
- Python 3.9+
- CUDA 11.0+ (for GPU support)
- Conda (recommended)

### Required Packages
```
gymnasium>=1.0.0
gymnasium-robotics>=1.3.0
minari>=0.4.0
torch>=2.0.0
numpy
einops
matplotlib
tqdm
pyyaml
```

### Install from requirements.txt
```bash
pip install -r requirements.txt
```

### Troubleshooting

**MuJoCo Version Conflicts:**
```bash
pip uninstall dm-control
```

**Dataset Download Issues:**
```bash
pip install minari --upgrade
```

**CUDA Out of Memory:**
```bash
# Reduce batch size or horizon
python scripts/train.py --batch-size 128 --horizon 32
```

---

### Step 2: Download Data
```bash
# List available datasets
python scripts/download_data.py --list

# Download PointMaze dataset (recommended for quick testing)
python scripts/download_data.py --dataset D4RL/pointmaze/umaze-v2

# Check what's downloaded
python scripts/download_data.py --list-local
```

### Step 3: Train Model
```bash
# Train on PointMaze (fast, ~30 min on GPU)
python scripts/train.py \
    --dataset D4RL/pointmaze/umaze-v2 \
    --horizon 32 \
    --dim 128 \
    --dim-mults 1 2 4 \
    --n-timesteps 100 \
    --batch-size 256 \
    --lr 6e-4 \
    --n-epochs 50 \
    --gradient-clip 5.0 \
    --use-ema \
    --device cuda
```

Model checkpoint saved to: `./logs/D4RL/pointmaze/umaze-v2/final_model.pt`

### Step 4: Evaluate

#### Vanilla Diffusion (Baseline)
```bash
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type guided \
    --n-episodes 10 \
    --sampling-timesteps 500 \
    --results-dir ./results/vanilla \
    --video-dir ./videos/vanilla \
    --render video \
    --seed 42 \
    --device cuda
```

#### Dynamics-Aware (Our Method)
```bash
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type dynamics-aware \
    --n-episodes 10 \
    --sampling-timesteps 500 \
    --results-dir ./results/dynamics \
    --video-dir ./videos/dynamics \
    --render video \
    --seed 42 \
    --device cuda
```

**Note:** Using the same `--seed 42` ensures both methods are evaluated on identical episodes for fair comparison.



## Datasets

### Download Commands
```bash
# Quick test (small, fast)
python scripts/download_data.py --dataset D4RL/pointmaze/umaze-v2

# Locomotion
python scripts/download_data.py --dataset D4RL/halfcheetah/medium-v2

# Manipulation (large)
python scripts/download_data.py --dataset D4RL/door/expert-v2
```

### Storage Location
- **Linux/Mac**: `~/.minari/datasets/`
- **Windows**: `C:\Users\<username>\.minari\datasets\`

---

## Training

#### PointMaze (Fast Training, Good for Testing)
```bash
python scripts/train.py \
    --dataset D4RL/pointmaze/umaze-v2 \
    --horizon 32 \
    --dim 128 \
    --dim-mults 1 2 4 \
    --n-timesteps 100 \
    --batch-size 256 \
    --lr 6e-4 \
    --n-epochs 50 \
    --gradient-clip 5.0 \
    --warmup-steps 2000 \
    --use-ema \
    --ema-decay 0.995 \
    --device cuda
```

#### HalfCheetah (Locomotion Benchmark)
```bash
python scripts/train.py \
    --dataset D4RL/halfcheetah/medium-v2 \
    --horizon 32 \
    --dim 256 \
    --dim-mults 1 4 8 \
    --n-timesteps 1000 \
    --batch-size 128 \
    --lr 2e-4 \
    --n-epochs 100 \
    --gradient-clip 5.0 \
    --warmup-steps 2000 \
    --use-ema \
    --device cuda
```

#### AdroitHand Door (Complex Manipulation)
```bash
python scripts/train.py \
    --dataset D4RL/door/expert-v2 \
    --horizon 32 \
    --dim 256 \
    --dim-mults 1 2 4 8 \
    --n-timesteps 1000 \
    --batch-size 128 \
    --lr 2e-4 \
    --n-epochs 100 \
    --gradient-clip 5.0 \
    --warmup-steps 2000 \
    --use-ema \
    --device cuda
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | - | Minari dataset name (required) |
| `--horizon` | 64 | Planning horizon length |
| `--dim` | 128 | Base U-Net dimension |
| `--dim-mults` | 1 2 4 8 | Dimension multipliers |
| `--n-timesteps` | 1000 | Diffusion training steps |
| `--batch-size` | 64 | Training batch size |
| `--lr` | 3e-4 | Learning rate |
| `--gradient-clip` | 1.0 | Gradient clipping (⚠️ use 5.0+) |
| `--warmup-steps` | 2000 | LR warmup steps |
| `--n-epochs` | 100 | Number of epochs |
| `--use-ema` | True | Use EMA |
| `--ema-decay` | 0.995 | EMA decay rate |
| `--device` | cuda | Device (cuda/cpu) |
| `--log-dir` | ./logs | Checkpoint directory |


### Monitoring Training

Checkpoints saved every epoch:
```
./logs/D4RL/pointmaze/umaze-v2/
├── config.json
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_20.pt
├── ...
└── final_model.pt
```

---

## Evaluation

### Policy Types

1. **`guided`** - Vanilla diffusion (baseline)
2. **`dynamics-aware`** - Our method with projection (🆕)
3. **`mpc`** - Model Predictive Control (replans every step)

### Basic Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type guided \
    --n-episodes 10 \
    --device cuda
```

### Dynamics-Aware Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type dynamics-aware \
    --n-episodes 10 \
    --sampling-timesteps 500 \
    --device cuda
```

**What happens:**
1. Extracts dynamics (A, B matrices) from environment
2. Builds projection matrix P = FF†
3. Applies projection at each denoising step
4. Ensures trajectories satisfy x_{t+1} = Ax_t + Bu_t

### With Video Recording
```bash
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type dynamics-aware \
    --render video \
    --video-dir ./videos/dynamics_aware \
    --results-dir ./results/dynamics_aware \
    --n-episodes 5 \
    --device cuda
```

Videos saved to: `./videos/dynamics_aware/rl-video-episode-*.mp4`

### Fair Comparison Setup
```bash
# 1. Vanilla baseline
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type guided \
    --results-dir ./results/vanilla \
    --video-dir ./videos/vanilla \
    --render video \
    --n-episodes 10 \
    --seed 42 \
    --device cuda

# 2. Dynamics-aware (SAME SEED!)
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type dynamics-aware \
    --results-dir ./results/dynamics \
    --video-dir ./videos/dynamics \
    --render video \
    --n-episodes 10 \
    --seed 42 \
    --device cuda

# 3. Compare
python scripts/compare_results.py \
    results/vanilla/guided_*.json \
    results/dynamics/dynamics_aware_*.json
```

### Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | - | Path to trained model (required) |
| `--env` | - | Gymnasium environment name |
| `--dataset` | - | Dataset name (for normalizer) |
| `--policy-type` | guided | Policy: guided, mpc, dynamics-aware |
| `--n-episodes` | 10 | Number of episodes |
| `--sampling-timesteps` | 100 | Inference diffusion steps (↓ = faster) |
| `--render` | none | Render: none, human, video |
| `--video-dir` | ./videos | Video save directory |
| `--results-dir` | ./results | Results JSON directory |
| `--seed` | 42 | Random seed |
| `--device` | cuda | Device (cuda/cpu) |

### Results Format

Results saved as JSON:
```json
{
  "policy_type": "dynamics-aware",
  "environment": "PointMaze_UMaze-v3",
  "n_episodes": 10,
  "metrics": {
    "mean_reward": -10.23,
    "std_reward": 1.85,
    "episode_rewards": [-9.5, -10.8, ...],
    "episode_lengths": [145, 158, ...]
  }
}
```

---

## 📁 Project Structure
```
m_diffuser/
├── m_diffuser/
│   ├── models/
│   │   ├── temporal_unet.py      # Temporal U-Net (from Janner et al., 2022)
│   │   └── diffusion.py          # Gaussian diffusion (DDPM)
│   ├── datasets/
│   │   ├── sequence.py           # Trajectory sequences
│   │   └── normalization.py      # Data normalization
│   ├── dynamics/                  # 🆕 Dynamics-aware components
│   │   ├── __init__.py
│   │   ├── extractor.py          # Extract (A, B) from environments
│   │   ├── projection.py         # Build projection matrices P = FF†
│   │   └── registry.py           # Environment → dynamics mapping
│   ├── guides/
│   │   └── policies.py           # Policies (guided, MPC, dynamics-aware)
│   └── utils/
│       ├── arrays.py             # Array utilities
│       └── training.py           # Training infrastructure
├── scripts/
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   ├── download_data.py          # Dataset downloader
├── config/                       # Config files
├── logs/                         # Training checkpoints
├── results/                      # Evaluation results (JSON)
├── videos/                       # Recorded videos
├── requirements.txt
└── README.md
```

---


## Usage Examples

### Example 1: Train and Evaluate on PointMaze
```bash
# Download data
python scripts/download_data.py --dataset D4RL/pointmaze/umaze-v2

# Train model
python scripts/train.py \
    --dataset D4RL/pointmaze/umaze-v2 \
    --n-epochs 50 \
    --device cuda

# Evaluate vanilla
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type guided \
    --n-episodes 10

# Evaluate dynamics-aware
python scripts/evaluate.py \
    --checkpoint logs/D4RL/pointmaze/umaze-v2/final_model.pt \
    --env PointMaze_UMaze-v3 \
    --dataset D4RL/pointmaze/umaze-v2 \
    --policy-type dynamics-aware \
    --n-episodes 10
```

### Example 2: Python API
```python
from m_diffuser.models import TemporalUnet, GaussianDiffusion
from m_diffuser.datasets import SequenceDataset
from m_diffuser.guides.policies import DynamicsAwarePolicy
from m_diffuser.dynamics import get_dynamics_for_env, ProjectionMatrixBuilder

# Load dataset
dataset = SequenceDataset('D4RL/pointmaze/umaze-v2', horizon=32)

# Create model (U-Net from Janner et al.)
unet = TemporalUnet(
    transition_dim=dataset.transition_dim,
    dim=128,
    dim_mults=(1, 2, 4)
)

diffusion = GaussianDiffusion(
    model=unet,
    horizon=32,
    observation_dim=dataset.observation_dim,
    action_dim=dataset.action_dim,
    n_timesteps=100
)

# Extract dynamics
A, B, state_dim, action_dim = get_dynamics_for_env('PointMaze_UMaze-v3')

# Build projection matrix
proj_builder = ProjectionMatrixBuilder(A, B, state_dim, action_dim)
P = proj_builder.get_projection_matrix(horizon=32)

# Create dynamics-aware policy
policy = DynamicsAwarePolicy(
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    projection_matrix=P,
    state_dim=state_dim,
    action_dim=action_dim
)

# Generate action
action = policy.get_action(observation)
```


## Citation

```
@article{gadginmath2025dynamics,
  title={Dynamics-aware Diffusion Models for Planning and Control},
  author={Gadginmath, Darshan and Pasqualetti, Fabio},
  journal={arXiv preprint arXiv:2504.00236},
  year={2025}
}
```

##  License

MIT License - see LICENSE file for details.

---

##  Resources

- **Original Diffuser**: [https://diffusion-planning.github.io/](https://diffusion-planning.github.io/)
- **Original Code**: [https://github.com/jannerm/diffuser](https://github.com/jannerm/diffuser)
- **Minari**: [https://minari.farama.org/](https://minari.farama.org/)
- **Gymnasium**: [https://gymnasium.farama.org/](https://gymnasium.farama.org/)
- **Dynamics Paper**: [arXiv:2504.00236](https://arxiv.org/abs/2504.00236)

---

##  Acknowledgments

- **Temporal U-Net architecture** adapted from Janner et al. (2022)
- **Dynamics-aware projection** based on Gadginmath & Pasqualetti (2025)
- Built on **Gymnasium**, **Minari**, and **PyTorch** ecosystems
