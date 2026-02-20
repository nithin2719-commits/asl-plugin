"""
Utility functions for Sign Language Recognition training.

This module provides:
- Accuracy calculation functions
- AverageMeter for tracking metrics
- Checkpoint saving/loading utilities
- Early stopping implementation
"""

import torch
from typing import Tuple, Optional
import numpy as np


def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate top-1 accuracy.
    
    Args:
        outputs: Model predictions of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size,)
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    with torch.no_grad():
        _, predicted = torch.max(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
    return accuracy


class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
        verbose: Print message when early stopping triggers
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric (loss or accuracy)
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # Check if score improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training. Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
                
        return False
    
    def reset(self):
        """Reset the early stopping counter."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0


def calculate_topk_accuracy(
    outputs: torch.Tensor, 
    targets: torch.Tensor, 
    topk: Tuple[int, ...] = (1, 5)
) -> list:
    """
    Calculate top-k accuracy for multiple k values.
    
    Args:
        outputs: Model predictions of shape (batch_size, num_classes)
        targets: Ground truth labels of shape (batch_size,)
        topk: Tuple of k values for top-k accuracy
    
    Returns:
        List of accuracies for each k value
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        
        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(1.0 / batch_size).item()
            accuracies.append(accuracy)
        
        return accuracies


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.
        
        Args:
            val: New value to add
            n: Number of samples this value represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    state: dict,
    filepath: str
) -> None:
    """
    Save model checkpoint to file.
    
    Args:
        state: Dictionary containing model state and metadata
        filepath: Path to save the checkpoint
    """
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> dict:
    """
    Load model checkpoint from file.
    
    Args:
        filepath: Path to the checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint
