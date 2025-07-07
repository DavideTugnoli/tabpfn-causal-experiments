"""
Checkpoint utilities for experiment resumption.

This module provides common checkpointing functionality used across
all experiments to save and restore progress.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def save_checkpoint(results_so_far: List[Dict[str, Any]], 
                   current_train_idx: int, 
                   current_rep: int, 
                   output_dir: Path,
                   checkpoint_name: str = "checkpoint.pkl") -> None:
    """
    Save experiment checkpoint.
    
    Args:
        results_so_far: List of results collected so far
        current_train_idx: Current training size index
        current_rep: Current repetition index
        output_dir: Output directory path
        checkpoint_name: Name of checkpoint file
    """
    checkpoint = {
        'results': results_so_far,
        'current_train_idx': current_train_idx,
        'current_rep': current_rep
    }
    
    checkpoint_file = output_dir / checkpoint_name
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(output_dir: Path, 
                   checkpoint_name: str = "checkpoint.pkl") -> Optional[Dict[str, Any]]:
    """
    Load experiment checkpoint.
    
    Args:
        output_dir: Output directory path
        checkpoint_name: Name of checkpoint file
        
    Returns:
        Checkpoint data if exists, None otherwise
    """
    checkpoint_file = output_dir / checkpoint_name
    if checkpoint_file.exists():
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)
    return None


def cleanup_checkpoint(output_dir: Path, 
                      checkpoint_name: str = "checkpoint.pkl") -> None:
    """
    Remove checkpoint file after successful completion.
    
    Args:
        output_dir: Output directory path
        checkpoint_name: Name of checkpoint file
    """
    checkpoint_file = output_dir / checkpoint_name
    if checkpoint_file.exists():
        checkpoint_file.unlink()


def get_checkpoint_info(output_dir: Path, 
                       checkpoint_name: str = "checkpoint.pkl") -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Load checkpoint and return resume information.
    
    Args:
        output_dir: Output directory path
        checkpoint_name: Name of checkpoint file
        
    Returns:
        Tuple of (results_so_far, start_train_idx, start_rep)
    """
    results_so_far = []
    start_train_idx = 0
    start_rep = 0
    
    checkpoint = load_checkpoint(output_dir, checkpoint_name)
    if checkpoint:
        print("Resuming from checkpoint!")
        results_so_far = checkpoint['results']
        start_train_idx = checkpoint['current_train_idx']
        start_rep = checkpoint['current_rep']
        print(f"  Resuming from train_size_idx={start_train_idx}, rep={start_rep}")
    
    return results_so_far, start_train_idx, start_rep 