# 🚀 Quick Start Guide

Get up and running with Modern Diffuser in 5 minutes!

## Step 1: Setup Environment

```bash
# Create conda environment
conda create -n modern_diffuser python=3.9
conda activate modern_diffuser

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Download Data

```bash
# Download common locomotion datasets
python scripts/download_data.py

# Or download a specific dataset
python scripts/download_data.py --dataset halfcheetah-medium-v0
```

## Step 3: Train a Model

```bash
# Basic training (HalfCheetah)
python scripts/train.py --dataset halfcheetah-medium-v0 --n-epochs 50

# Training will save checkpoints to ./logs/halfcheetah-medium-v0/
```

**Expected output:**
```
Loading dataset: halfcheetah-medium-v0
Dataset loaded: 1000 trajectories, 32000 sequences
Model parameters: 12,345,678
Starting training...
Epoch 0 | Step 100 | Loss: 0.0523
...
```

## Step 4: Evaluate

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint ./logs/halfcheetah-medium-v0/final_model.pt \
    --env HalfCheetah-v5 \
    --n-episodes 10
```

**Expected output:**
```
Evaluating Diffuser Model
Episode 1: Reward = 4532.15, Length = 1000
Episode 2: Reward = 4621.33, Length = 1000
...
Mean Reward: 4576.24 ± 124.56
```

## Step 5: Visualize (Optional)

```bash
# Evaluate with rendering
python scripts/evaluate.py \
    --checkpoint ./logs/halfcheetah-medium-v0/final_model.pt \
    --env HalfCheetah-v5 \
    --render
```

---

## Common Commands

### Training Variations

```bash
# Longer horizon
python scripts/train.py --dataset halfcheetah-medium-v0 --horizon 128

# Smaller model (faster)
python scripts/train.py --dataset halfcheetah-medium-v0 --dim 64 --dim-mults 1 2 4

# CPU training
python scripts/train.py --dataset halfcheetah-medium-v0 --device cpu
```

### Different Environments

```bash
# Hopper
python scripts/train.py --dataset hopper-medium-v0

# Walker2d
python scripts/train.py --dataset walker2d-medium-v0
```

### Policy Types

```bash
# Standard guided policy
python scripts/evaluate.py --checkpoint MODEL.pt --env ENV --policy-type guided

# MPC (replans every step, slower but better)
python scripts/evaluate.py --checkpoint MODEL.pt --env ENV --policy-type mpc --action-horizon 8
```

---

## Typical Training Time

On a modern GPU (RTX 3080/4090):
- **HalfCheetah**: ~2-3 hours for 100 epochs
- **Hopper**: ~1-2 hours for 100 epochs
- **Walker2d**: ~2-3 hours for 100 epochs

On CPU: 10-20x slower

---

## Troubleshooting

### Issue: Dataset not found
```bash
# List available datasets
python scripts/download_data.py --list

# Download specific dataset
python scripts/download_data.py --dataset DATASET_NAME
```

### Issue: CUDA out of memory
```bash
# Reduce batch size
python scripts/train.py --batch-size 16

# Reduce horizon
python scripts/train.py --horizon 32
```

### Issue: MuJoCo version conflict
```bash
# Remove dm-control if not needed
pip uninstall dm-control
```

---

## Next Steps

1. **Experiment with hyperparameters** - Edit config files in `config/experiments/`
2. **Try different datasets** - See available datasets with `--list`
3. **Implement custom guidance** - Modify `m_diffuser/guides/policies.py`
4. **Add your own environment** - Extend dataset loading in `m_diffuser/datasets/`

---

## File Structure Reference

```
m_diffuser/
├── scripts/
│   ├── train.py              # <- Start here
│   ├── evaluate.py           # <- Then here
│   └── download_data.py      # <- Data management
├── m_diffuser/
│   ├── models/               # Core models
│   ├── datasets/             # Data loading
│   ├── guides/               # Planning policies
│   └── utils/                # Utilities
├── config/
│   └── experiments/          # Config files
└── logs/                     # Training outputs
```

---

## Help & Resources

- **Check dataset info**: `python