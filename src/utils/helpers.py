"""
Utility Helper Functions
"""

import yaml
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        format_string: Custom format string
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {level} level")
    
    return logger


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    logging.getLogger(__name__).info(f"Random seed set to {seed}")


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_project_root() -> Path:
    """
    Get the project root directory
    
    Returns:
        Path to project root
    """
    current = Path(__file__).resolve()
    
    # Navigate up to find project root (contains config/)
    for parent in current.parents:
        if (parent / "config").exists():
            return parent
    
    # Default to current working directory
    return Path.cwd()


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format large numbers with commas
    
    Args:
        num: Number to format
        decimals: Decimal places
        
    Returns:
        Formatted string
    """
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:,.{decimals}f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:,.{decimals}f}K"
    else:
        return f"{num:,.{decimals}f}"


def calculate_class_weights(y, method: str = 'balanced') -> Dict[int, float]:
    """
    Calculate class weights for imbalanced data
    
    Args:
        y: Target labels
        method: 'balanced' or 'sqrt'
        
    Returns:
        Dictionary of class weights
    """
    from collections import Counter
    
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    
    if method == 'balanced':
        weights = {
            cls: total / (n_classes * count)
            for cls, count in counts.items()
        }
    elif method == 'sqrt':
        weights = {
            cls: np.sqrt(total / count)
            for cls, count in counts.items()
        }
    else:
        weights = {cls: 1.0 for cls in counts.keys()}
    
    return weights
