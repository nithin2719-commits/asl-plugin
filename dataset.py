"""
Custom Dataset class for loading WLASL sign language videos.

This module provides:
- WLASLDataset: PyTorch Dataset for video loading and preprocessing
- LabelEncoder integration for automatic label encoding
- Support for both decord and torchvision video backends
- Uniform frame sampling with padding for short videos
"""

import os
from typing import Tuple, Optional, Callable
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Try to import decord (faster), fallback to torchvision
try:
    from decord import VideoReader, cpu
    USE_DECORD = True
except ImportError:
    USE_DECORD = False
    import torchvision.io as video_io


# ImageNet normalization parameters (used for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WLASLDataset(Dataset):
    """
    PyTorch Dataset for WLASL sign language video recognition.
    
    Handles video loading, frame sampling, resizing, and normalization.
    Uses LabelEncoder for automatic encoding of class labels.
    
    Args:
        root_dir: Root directory containing the dataset (with videos/ subfolder)
        annotation_file: Path to annotation file (train.txt, val.txt, or test.txt)
        num_frames: Number of frames to uniformly sample from each video
        frame_size: Target size for frames (height, width)
        transform: Optional additional transforms to apply
        label_encoder: Pre-fitted LabelEncoder (use for val/test sets)
    
    Output tensor shape: (C, T, H, W) = (3, num_frames, height, width)
    """
    
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        num_frames: int = 32,
        frame_size: Tuple[int, int] = (224, 224),
        transform: Optional[Callable] = None,
        label_encoder: Optional[LabelEncoder] = None
    ):
        self.root_dir = root_dir
        candidate_video_dir = os.path.join(root_dir, "videos")
        if os.path.isdir(candidate_video_dir):
            self.video_dir = candidate_video_dir
        else:
            parent_video_dir = os.path.join(os.path.dirname(root_dir), "videos")
            self.video_dir = parent_video_dir if os.path.isdir(parent_video_dir) else candidate_video_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform
        
        # Load annotations from file
        self.samples = []  # Video file paths
        self.labels = []   # Original labels (before encoding)
        self._load_annotations(annotation_file)
        
        # Setup LabelEncoder for converting labels to indices
        if label_encoder is not None:
            # Use provided encoder (for val/test consistency with training)
            self.label_encoder = label_encoder
            known_labels = set(int(v) for v in self.label_encoder.classes_.tolist())
            filtered_samples = []
            filtered_labels = []
            for sample_path, label in zip(self.samples, self.labels):
                if int(label) in known_labels:
                    filtered_samples.append(sample_path)
                    filtered_labels.append(label)
            self.samples = filtered_samples
            self.labels = filtered_labels
        else:
            # Create and fit new encoder (for training set)
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.labels)
        
        # Encode all labels to indices
        self.encoded_labels = self.label_encoder.transform(self.labels)
        self.num_classes = len(self.label_encoder.classes_)
        
        # Precompute normalization tensors for efficiency
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1, 1)
        
        print(f"Loaded {len(self.samples)} videos with {self.num_classes} classes")
        print(f"Video backend: {'decord' if USE_DECORD else 'torchvision'}")
    
    def _load_annotations(self, annotation_file: str) -> None:
        """
        Load video paths and labels from annotation file.
        
        Expected format per line: video_name.mp4 label
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) >= 2:
                    video_name = parts[0]
                    label = int(parts[1])
                    video_path = os.path.join(self.video_dir, video_name)
                    
                    # Only add if video file exists
                    if os.path.exists(video_path):
                        self.samples.append(video_path)
                        self.labels.append(label)
                    else:
                        print(f"Warning: Video not found: {video_path}")
    
    def _sample_frame_indices(self, total_frames: int) -> np.ndarray:
        """
        Sample frame indices uniformly from the video.
        
        If video has fewer frames than required, pad by repeating last frame.
        
        Args:
            total_frames: Total number of frames in the video
        
        Returns:
            Array of frame indices to sample (length = self.num_frames)
        """
        if total_frames >= self.num_frames:
            # Uniform sampling across video duration
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=np.int64)
        else:
            # Video too short - use all frames and pad with last frame
            indices = np.arange(total_frames)
            pad_length = self.num_frames - total_frames
            pad_indices = np.full(pad_length, total_frames - 1, dtype=np.int64)
            indices = np.concatenate([indices, pad_indices])
        
        return indices
    
    def _load_video_decord(self, video_path: str) -> torch.Tensor:
        """
        Load video frames using decord.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Tensor of shape (C, T, H, W)
        """
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        indices = self._sample_frame_indices(total_frames)
        frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        
        # Convert to tensor and resize
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        frames = torch.nn.functional.interpolate(
            frames, 
            size=self.frame_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1] then apply ImageNet normalization
        frames = frames / 255.0
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        frames = (frames - self.mean) / self.std
        
        return frames
    
    def _load_video_torchvision(self, video_path: str) -> torch.Tensor:
        """
        Load video frames using torchvision.
        
        Args:
            video_path: Path to the video file
        
        Returns:
            Tensor of shape (C, T, H, W)
        """
        # Read video
        video, audio, info = video_io.read_video(video_path, pts_unit='sec')
        # video shape: (T, H, W, C)
        
        total_frames = video.shape[0]
        if total_frames == 0:
            # Return empty tensor with correct shape
            return torch.zeros(3, self.num_frames, *self.frame_size)
        
        indices = self._sample_frame_indices(total_frames)
        frames = video[indices]  # (T, H, W, C)
        
        # Convert to float and resize
        frames = frames.permute(0, 3, 1, 2).float()  # (T, C, H, W)
        frames = torch.nn.functional.interpolate(
            frames, 
            size=self.frame_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1] then apply ImageNet normalization
        frames = frames / 255.0
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        frames = (frames - self.mean) / self.std
        
        return frames
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a video sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple of (video_tensor, label)
            video_tensor shape: (C, T, H, W) = (3, 32, 224, 224)
        """
        video_path = self.samples[idx]
        
        # Load video frames
        try:
            if USE_DECORD:
                video = self._load_video_decord(video_path)
            else:
                video = self._load_video_torchvision(video_path)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return zero tensor on error
            video = torch.zeros(3, self.num_frames, *self.frame_size)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            video = self.transform(video)
        
        # Get encoded label (already transformed by LabelEncoder)
        label_idx = self.encoded_labels[idx]
        
        return video, label_idx
    
    def get_num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return self.num_classes
    
    def get_label_encoder(self) -> LabelEncoder:
        """Return the fitted LabelEncoder."""
        return self.label_encoder
    
    def save_label_encoder(self, filepath: str) -> None:
        """Save the LabelEncoder to a file for inference."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"LabelEncoder saved to {filepath}")
    
    @staticmethod
    def load_label_encoder(filepath: str) -> LabelEncoder:
        """Load a saved LabelEncoder."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def get_num_classes_from_file(annotation_file: str) -> int:
    """
    Get the number of unique classes from an annotation file.
    
    Args:
        annotation_file: Path to annotation file
    
    Returns:
        Number of unique classes
    """
    labels = set()
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                labels.add(int(parts[1]))
    return len(labels)


def create_data_loaders(
    root_dir: str,
    train_file: str,
    val_file: str,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 32,
    frame_size: Tuple[int, int] = (224, 224)
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, int, LabelEncoder]:
    """
    Create training and validation data loaders with shared LabelEncoder.
    
    Args:
        root_dir: Root directory containing the dataset
        train_file: Path to training annotation file
        val_file: Path to validation annotation file
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        num_frames: Number of frames to sample per video
        frame_size: Size to resize frames to
    
    Returns:
        Tuple of (train_loader, val_loader, num_classes, label_encoder)
    """
    # Create training dataset (fits the LabelEncoder)
    train_dataset = WLASLDataset(
        root_dir=root_dir,
        annotation_file=train_file,
        num_frames=num_frames,
        frame_size=frame_size,
        label_encoder=None  # Will create and fit new encoder
    )
    
    # Create validation dataset (uses training LabelEncoder)
    val_dataset = WLASLDataset(
        root_dir=root_dir,
        annotation_file=val_file,
        num_frames=num_frames,
        frame_size=frame_size,
        label_encoder=train_dataset.label_encoder  # Share encoder
    )
    
    # Create data loaders with optimized settings
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, train_dataset.num_classes, train_dataset.label_encoder


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WLASL Dataset")
    parser.add_argument("--root_dir", type=str, default="dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--train_file", type=str, default="dataset/train.txt",
                        help="Path to training annotation file")
    args = parser.parse_args()
    
    print("Testing WLASLDataset...")
    print("-" * 50)
    
    dataset = WLASLDataset(
        root_dir=args.root_dir,
        annotation_file=args.train_file
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Classes: {dataset.label_encoder.classes_[:10]}...")  # Show first 10
    
    if len(dataset) > 0:
        video, label = dataset[0]
        print(f"\nSample video shape: {video.shape}")
        print(f"Expected shape: (3, 32, 224, 224)")
        print(f"Sample label index: {label}")
        print(f"Original label: {dataset.label_encoder.inverse_transform([label])[0]}")
