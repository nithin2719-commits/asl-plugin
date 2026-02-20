"""
I3D Model wrapper for Sign Language Recognition.

This module provides:
- I3DModel: Wrapper around pretrained 3D CNNs from torchvision
- Supports R3D-18, MC3-18, R(2+1)D-18 backbones (all pretrained on Kinetics-400)
- Automatic classification head replacement with dropout
- Options to freeze backbone for transfer learning

The models use "inflated" 3D convolutions that extend 2D convolutional weights
into the temporal dimension, allowing them to learn spatiotemporal features.
"""

import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.models.video import mc3_18, MC3_18_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

# Try to import MViT (available in newer torchvision versions)
try:
    from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
    MVIT_AVAILABLE = True
except ImportError:
    MVIT_AVAILABLE = False


class I3DModel(nn.Module):
    """
    I3D-style model for video classification using ResNet3D-18 backbone.
    
    This uses a 3D ResNet pretrained on Kinetics-400 as an "I3D-like" model,
    which achieves similar performance with better PyTorch integration.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        backbone: Backbone to use ('r3d_18', 'mc3_18', 'r2plus1d_18', 'mvit_v2_s')
        dropout: Dropout probability before the final classifier
        freeze_backbone: Whether to freeze backbone layers initially
    """
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        backbone: str = 'r3d_18',
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # Load pretrained backbone
        if backbone == 'r3d_18':
            if pretrained:
                weights = R3D_18_Weights.KINETICS400_V1
                self.backbone = r3d_18(weights=weights)
            else:
                self.backbone = r3d_18(weights=None)
            in_features = self.backbone.fc.in_features
            
        elif backbone == 'mc3_18':
            if pretrained:
                weights = MC3_18_Weights.KINETICS400_V1
                self.backbone = mc3_18(weights=weights)
            else:
                self.backbone = mc3_18(weights=None)
            in_features = self.backbone.fc.in_features
            
        elif backbone == 'r2plus1d_18':
            if pretrained:
                weights = R2Plus1D_18_Weights.KINETICS400_V1
                self.backbone = r2plus1d_18(weights=weights)
            else:
                self.backbone = r2plus1d_18(weights=None)
            in_features = self.backbone.fc.in_features
            
        elif backbone == 'mvit_v2_s' and MVIT_AVAILABLE:
            if pretrained:
                weights = MViT_V2_S_Weights.KINETICS400_V1
                self.backbone = mvit_v2_s(weights=weights)
            else:
                self.backbone = mvit_v2_s(weights=None)
            in_features = self.backbone.head[1].in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Choose from 'r3d_18', 'mc3_18', 'r2plus1d_18'"
                           + (", 'mvit_v2_s'" if MVIT_AVAILABLE else ""))
        
        # Replace the final classification layer
        if backbone == 'mvit_v2_s' and MVIT_AVAILABLE:
            self.backbone.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self) -> None:
        """Freeze all backbone layers except the final classifier."""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name and 'head' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B = batch size
               C = channels (3)
               T = temporal frames (32)
               H, W = spatial dimensions (224, 224)
        
        Returns:
            Output tensor of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_trainable_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Return the total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_model(
    num_classes: int,
    pretrained: bool = True,
    backbone: str = 'r3d_18',
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    device: torch.device = None
) -> I3DModel:
    """
    Factory function to create an I3D model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        backbone: Backbone architecture to use
        dropout: Dropout probability
        freeze_backbone: Whether to freeze backbone layers
        device: Device to place the model on
    
    Returns:
        I3DModel instance
    """
    model = I3DModel(
        num_classes=num_classes,
        pretrained=pretrained,
        backbone=backbone,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    
    if device is not None:
        model = model.to(device)
    
    print(f"Created {backbone} model with {num_classes} classes")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    
    return model


class I3DWithFeatures(I3DModel):
    """
    Extended I3D model that can also return intermediate features.
    Useful for visualization or transfer learning.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.features = None
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass with optional feature extraction.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
            return_features: Whether to return intermediate features
        
        Returns:
            If return_features is False: Output tensor of shape (B, num_classes)
            If return_features is True: Tuple of (output, features)
        """
        if not return_features:
            return self.backbone(x)
        
        # Extract features before final classifier
        if self.backbone_name == 'mvit_v2_s':
            # For MViT, we need custom feature extraction
            features = self._extract_features_mvit(x)
        else:
            features = self._extract_features_r3d(x)
        
        self.features = features
        
        # Apply classifier
        if self.backbone_name == 'mvit_v2_s':
            output = self.backbone.head(features)
        else:
            output = self.backbone.fc(features)
        
        return output, features
    
    def _extract_features_r3d(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from R3D-style backbone."""
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = x.flatten(1)
        return x
    
    def _extract_features_mvit(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from MViT backbone."""
        # This is a simplified version - full implementation would 
        # require accessing MViT's internal structure
        x = self.backbone.conv_proj(x)
        # Continue through transformer blocks...
        # This is backbone-specific
        return x


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test with different backbones
    for backbone in ['r3d_18', 'mc3_18', 'r2plus1d_18']:
        print(f"\nTesting {backbone}:")
        model = create_model(
            num_classes=100,
            pretrained=True,
            backbone=backbone,
            device=device
        )
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 3, 32, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
