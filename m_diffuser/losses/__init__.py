"""
m_diffuser/losses/__init__.py

Modular loss system for composing different training objectives.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseLoss(ABC, nn.Module):
    """Base class for all loss functions."""
    
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight
    
    @abstractmethod
    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch.
        
        Args:
            batch: Dictionary containing 'conditions' and other keys
        
        Returns:
            Scalar loss tensor
        """
        pass
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with weight applied."""
        return self.weight * self.compute(batch)


class DiffusionLoss(BaseLoss):
    """Standard diffusion model training loss."""
    
    def __init__(self, diffusion_model, weight: float = 1.0):
        super().__init__(weight)
        self.diffusion = diffusion_model
    
    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute standard diffusion loss."""
        trajectories = batch['conditions']
        return self.diffusion.loss(trajectories)


class ProjectionLoss(BaseLoss):
    """
    Dynamics projection loss.
    
    Penalizes trajectories that violate system dynamics by measuring
    distance from the dynamics-feasible subspace.
    
    IMPORTANT: Projection happens in PHYSICAL (unnormalized) space!
    """
    
    def __init__(self,
                 projection_matrix: torch.Tensor,
                 normalizer: 'DatasetNormalizer',  # ADD THIS!
                 state_dim: int,
                 action_dim: int,
                 observation_dim: int,
                 horizon: int,
                 weight: float = 0.1,
                 device: str = 'cuda'):
        super().__init__(weight)
        
        self.P = projection_matrix.to(device)
        self.normalizer = normalizer  # Store normalizer!
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.horizon = horizon
        self.device = device
        
        # Extract normalization statistics as tensors (for efficiency)
        self.obs_mean = torch.from_numpy(normalizer.obs_mean).float().to(device)
        self.obs_std = torch.from_numpy(normalizer.obs_std).float().to(device)
        self.action_mean = torch.from_numpy(normalizer.action_mean).float().to(device)
        self.action_std = torch.from_numpy(normalizer.action_std).float().to(device)
        
        print(f"ProjectionLoss initialized:")
        print(f"  Weight: {weight}")
        print(f"  State dim: {state_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Observation dim: {observation_dim}")
        print(f"  Horizon: {horizon}")
        print(f"  Projection matrix shape: {self.P.shape}")
    
    def extract_state_actions(self, trajectory: torch.Tensor):
        """
        Extract states and actions from interleaved trajectory.
        
        Args:
            trajectory: (batch, horizon, transition_dim) - NORMALIZED
        
        Returns:
            state: (batch, horizon, state_dim) - NORMALIZED
            actions: (batch, horizon, action_dim) - NORMALIZED
        """
        batch_size, horizon, transition_dim = trajectory.shape
        
        # Split observation and action (interleaved format)
        observations = trajectory[:, :, :self.observation_dim]  # (batch, horizon, 4)
        actions = trajectory[:, :, self.observation_dim:]        # (batch, horizon, 2)
        
        return observations, actions
    
    def unnormalize_states(self, states_norm: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize states from normalized to physical space.
        
        Args:
            states_norm: (batch, horizon, state_dim) - normalized
        
        Returns:
            states_physical: (batch, horizon, state_dim) - physical
        """
        return states_norm * self.obs_std + self.obs_mean
    
    def unnormalize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize actions from normalized to physical space.
        
        Args:
            actions_norm: (batch, horizon, action_dim) - normalized
        
        Returns:
            actions_physical: (batch, horizon, action_dim) - physical
        """
        return actions_norm * self.action_std + self.action_mean
    
    def to_concatenated(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Convert to concatenated format for projection.
        
        Concatenated: [s0, s1, ..., sT, a0, a1, ..., a_{T-1}]
        
        Args:
            state: (batch, horizon, state_dim)
            actions: (batch, horizon, action_dim)
        
        Returns:
            concat: (batch, (horizon+1)*state_dim + horizon*action_dim)
        """
        batch_size, horizon, _ = state.shape
        
        # Add final state (duplicate last state as approximation)
        state_extended = torch.cat([state, state[:, -1:, :]], dim=1)  # (batch, horizon+1, state_dim)
        
        # Flatten to concatenated format
        state_flat = state_extended.reshape(batch_size, -1)    # (batch, (horizon+1)*state_dim)
        actions_flat = actions.reshape(batch_size, -1)          # (batch, horizon*action_dim)
        
        return torch.cat([state_flat, actions_flat], dim=1)     # (batch, (horizon+1)*state_dim + horizon*action_dim)
    
    def compute(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute projection loss: ||τ - P·τ||² in PHYSICAL space
        
        This measures how much the trajectory violates dynamics.
        Lower is better (trajectory closer to dynamics-feasible subspace).
        """
        trajectories = batch['conditions']  # (batch, horizon, transition_dim) - NORMALIZED
        
        # Extract state and actions (still normalized)
        states_norm, actions_norm = self.extract_state_actions(trajectories)
        
        # UNNORMALIZE to physical space (CRITICAL!)
        states_physical = self.unnormalize_states(states_norm)
        actions_physical = self.unnormalize_actions(actions_norm)
        
        # Convert to concatenated format (physical space)
        concat_physical = self.to_concatenated(states_physical, actions_physical)
        
        # Project onto dynamics subspace (in physical space!)
        projected = concat_physical @ self.P
        
        # Measure violation (distance from feasible subspace)
        violation = torch.mean((concat_physical - projected) ** 2)
        
        return violation


class ComposedLoss(nn.Module):
    """
    Composes multiple loss functions.
    
    Usage:
        loss_fn = ComposedLoss([
            DiffusionLoss(diffusion, weight=1.0),
            ProjectionLoss(P, weight=0.1),
        ])
        
        total_loss, loss_dict = loss_fn(batch)
    """
    
    def __init__(self, losses: List[BaseLoss]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
    
    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Compute all losses and return total + breakdown.
        
        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary with individual loss values (for logging)
        """
        total_loss = 0.0
        loss_dict = {}
        
        for i, loss_fn in enumerate(self.losses):
            loss_value = loss_fn(batch)
            total_loss = total_loss + loss_value
            
            # Store for logging (detached)
            loss_name = loss_fn.__class__.__name__.replace('Loss', '').lower()
            loss_dict[loss_name] = loss_value.detach().item()
        
        loss_dict['total'] = total_loss.detach().item()
        
        return total_loss, loss_dict


# Export all loss classes
__all__ = [
    'BaseLoss',
    'DiffusionLoss',
    'ProjectionLoss',
    'ComposedLoss',
]