"""
Training script for Sign Language Recognition using I3D.

This script provides:
- Full training pipeline with validation
- Mixed precision training (AMP) for faster training on RTX 3070Ti
- Early stopping to prevent overfitting
- Checkpoint saving (best and last models)
- LabelEncoder saving for inference
"""

import os
import argparse
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import WLASLDataset, create_data_loaders, get_num_classes_from_file
from model import create_model
from utils import (
    calculate_accuracy, 
    AverageMeter, 
    save_checkpoint, 
    count_parameters,
    EarlyStopping
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train I3D for Sign Language Recognition'
    )
    
    # Data arguments
    parser.add_argument('--root_dir', type=str, default='dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--train_file', type=str, default='dataset/train.txt',
                        help='Path to training annotation file')
    parser.add_argument('--val_file', type=str, default='dataset/val.txt',
                        help='Path to validation annotation file')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='r3d_18',
                        choices=['r3d_18', 'mc3_18', 'r2plus1d_18'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='Freeze backbone for initial training')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Video arguments
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames to sample per video')
    parser.add_argument('--frame_size', type=int, default=224,
                        help='Frame size (height and width)')
    
    # Mixed precision
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (epochs without improvement)')
    
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    scaler: GradScaler = None
) -> tuple:
    """
    Train the model for one epoch.
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Accuracy')
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (videos, labels) in enumerate(pbar):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast():
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        acc = calculate_accuracy(outputs, labels)
        
        # Update meters
        batch_size = videos.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })
    
    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = True
) -> tuple:
    """
    Validate the model.
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Accuracy')
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for videos, labels in pbar:
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs, labels)
            
            # Update meters
            batch_size = videos.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(acc, batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.4f}'
            })
    
    return loss_meter.avg, acc_meter.avg


def main():
    """Main training function."""
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader, num_classes, label_encoder = create_data_loaders(
        root_dir=args.root_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
        frame_size=(args.frame_size, args.frame_size)
    )
    
    # Save LabelEncoder for inference
    label_encoder_path = os.path.join(args.save_dir, 'label_encoder.pkl')
    import pickle
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder saved to {label_encoder_path}")
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=num_classes,
        pretrained=args.pretrained,
        backbone=args.backbone,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        device=device
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            print(f"Resumed from epoch {start_epoch-1}, best val acc: {best_val_acc:.4f}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode='max',  # Monitoring validation accuracy
        verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp,
            scaler=scaler
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            use_amp=args.use_amp
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'num_classes': num_classes,
                    'backbone': args.backbone,
                },
                filepath=best_model_path
            )
            print(f"  New best model saved! Val Acc: {val_acc:.4f}")
        
        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.save_dir, 
                f'checkpoint_epoch_{epoch}.pth'
            )
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                    'num_classes': num_classes,
                    'backbone': args.backbone,
                },
                filepath=checkpoint_path
            )
        
        print("-" * 60)
        
        # Check early stopping
        if early_stopping(val_acc, epoch):
            print(f"\nEarly stopping triggered after {epoch} epochs!")
            break
    
    # Save last model
    last_model_path = os.path.join(args.save_dir, 'last_model.pth')
    save_checkpoint(
        state={
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'num_classes': num_classes,
            'backbone': args.backbone,
        },
        filepath=last_model_path
    )
    
    total_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")
    print(f"Last model saved to: {last_model_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
