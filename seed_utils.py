import random
import numpy as np
import torch
import os

def set_seed(seed, deterministic=True):
    """
    Set random seed for reproducibility across all modules
    
    Args:
        seed: Integer seed value
        deterministic: Whether to enable deterministic algorithms in PyTorch
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # Set seed for CUDA operations if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in PyTorch
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed} (deterministic={deterministic})")