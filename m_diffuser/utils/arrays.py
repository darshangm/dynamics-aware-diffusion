"""
Array manipulation utilities for diffusion planning.
"""

import numpy as np
import torch
from typing import Union, Optional


def to_torch(x: Union[np.ndarray, torch.Tensor], dtype=None, device='cuda') -> torch.Tensor:
    """
    Convert numpy array or torch tensor to torch tensor on specified device.
    
    Args:
        x: Input array or tensor
        dtype: Target dtype (e.g., torch.float32)
        device: Target device ('cuda' or 'cpu')
    
    Returns:
        Torch tensor on specified device
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x.to(device)
    else:
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(x, dtype=dtype, device=device)


def to_np(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert torch tensor to numpy array.
    
    Args:
        x: Input tensor or array
    
    Returns:
        Numpy array
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def to_device(x: torch.Tensor, device='cuda') -> torch.Tensor:
    """
    Move tensor to specified device.
    
    Args:
        x: Input tensor
        device: Target device
    
    Returns:
        Tensor on target device
    """
    if torch.cuda.is_available() and device == 'cuda':
        return x.cuda()
    return x.cpu()


def batch_to_device(batch: dict, device='cuda') -> dict:
    """
    Move all tensors in a batch dictionary to specified device.
    
    Args:
        batch: Dictionary containing tensors
        device: Target device
    
    Returns:
        Dictionary with tensors moved to device
    """
    return {
        key: to_device(val, device) if isinstance(val, torch.Tensor) else val
        for key, val in batch.items()
    }


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Normalize array using mean and standard deviation.
    
    Args:
        x: Input array
        mean: Mean for normalization
        std: Standard deviation for normalization
    
    Returns:
        Normalized array
    """
    return (x - mean) / (std + 1e-8)


def unnormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Unnormalize array using mean and standard deviation.
    
    Args:
        x: Normalized array
        mean: Mean used for normalization
        std: Standard deviation used for normalization
    
    Returns:
        Unnormalized array
    """
    return x * (std + 1e-8) + mean


def atleast_2d(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Ensure input has at least 2 dimensions.
    
    Args:
        x: Input array or tensor
    
    Returns:
        Array/tensor with at least 2 dimensions
    """
    if isinstance(x, torch.Tensor):
        while x.ndim < 2:
            x = x.unsqueeze(0)
    else:
        while x.ndim < 2:
            x = x[np.newaxis]
    return x


def apply_dict(fn, d: dict) -> dict:
    """
    Apply function to all values in dictionary.
    
    Args:
        fn: Function to apply
        d: Input dictionary
    
    Returns:
        Dictionary with function applied to values
    """
    return {k: fn(v) for k, v in d.items()}


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False