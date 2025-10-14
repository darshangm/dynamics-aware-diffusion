"""
Modern Diffuser: A PyTorch implementation of Diffuser for trajectory planning.

Based on "Planning with Diffusion for Flexible Behavior Synthesis" by Janner et al.
Adapted for Gymnasium and modern tooling.
"""

__version__ = "0.1.0"

from m_diffuser.models.temporal_unet import TemporalUnet
from m_diffuser.models.diffusion import GaussianDiffusion
from m_diffuser.datasets.sequence import SequenceDataset
from m_diffuser.guides.policies import GuidedPolicy, MPCPolicy
from m_diffuser.utils.training import Trainer

__all__ = [
    'TemporalUnet',
    'GaussianDiffusion',
    'SequenceDataset',
    'GuidedPolicy',
    'MPCPolicy',
    'Trainer',
]