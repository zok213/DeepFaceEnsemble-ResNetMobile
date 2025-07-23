"""
MobileFaceNet implementation for face recognition.
Based on: "MobileFaceNets: Efficient CNNs for Accurate Real-time Face Verification 
on Mobile Devices"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, 
                                   bias=bias)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        return x


class LinearBottleneck(nn.Module):
    """Linear Bottleneck Block for MobileFaceNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expand_ratio: int = 6, use_se: bool = True):
        super(LinearBottleneck, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True)
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])
        
        # SE block
        if use_se:
            layers.append(SEBlock(hidden_dim, reduction=4))
        
        # Project
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_residual:
            out += x
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MobileFaceNet(nn.Module):
    """MobileFaceNet Architecture."""
    
    def __init__(self, num_classes: int = 1000, embedding_dim: int = 512,
                 dropout: float = 0.5, width_multiplier: float = 1.0):
        super(MobileFaceNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Depthwise separable convolution
        self.conv2 = DepthwiseSeparableConv(64, 64, stride=1)
        
        # Bottleneck blocks
        self.bottleneck1 = LinearBottleneck(64, 64, stride=2, expand_ratio=2)
        self.bottleneck2 = LinearBottleneck(64, 64, stride=1, expand_ratio=2)
        self.bottleneck3 = LinearBottleneck(64, 64, stride=1, expand_ratio=2)
        self.bottleneck4 = LinearBottleneck(64, 64, stride=1, expand_ratio=2)
        self.bottleneck5 = LinearBottleneck(64, 64, stride=1, expand_ratio=2)
        
        self.bottleneck6 = LinearBottleneck(64, 128, stride=2, expand_ratio=4)
        self.bottleneck7 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        self.bottleneck8 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        self.bottleneck9 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        self.bottleneck10 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        self.bottleneck11 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        self.bottleneck12 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        
        self.bottleneck13 = LinearBottleneck(128, 128, stride=2, expand_ratio=4)
        self.bottleneck14 = LinearBottleneck(128, 128, stride=1, expand_ratio=2)
        
        # Final convolution
        self.conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        
        # Global pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Embedding layer
        self.embedding = nn.Linear(512, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Depthwise separable convolution
        x = self.conv2(x)
        
        # Bottleneck blocks
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.bottleneck8(x)
        x = self.bottleneck9(x)
        x = self.bottleneck10(x)
        x = self.bottleneck11(x)
        x = self.bottleneck12(x)
        
        x = self.bottleneck13(x)
        x = self.bottleneck14(x)
        
        # Final convolution
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Embedding
        embedding = self.embedding(x)
        embedding = self.bn_embedding(embedding)
        
        if return_embedding:
            return F.normalize(embedding, p=2, dim=1)
        
        # Classification
        logits = self.classifier(embedding)
        
        return logits, F.normalize(embedding, p=2, dim=1)


def mobilefacenet(num_classes: int = 1000, embedding_dim: int = 512,
                  dropout: float = 0.5, width_multiplier: float = 1.0,
                  pretrained: bool = False) -> MobileFaceNet:
    """MobileFaceNet model."""
    model = MobileFaceNet(num_classes=num_classes, embedding_dim=embedding_dim,
                         dropout=dropout, width_multiplier=width_multiplier)
    
    if pretrained:
        # Load pretrained weights if available
        try:
            # You would load pretrained weights here
            print("Warning: Pretrained weights not available for MobileFaceNet")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model


# Test function
if __name__ == "__main__":
    # Test MobileFaceNet
    model = mobilefacenet(num_classes=8631, embedding_dim=512)
    x = torch.randn(4, 3, 112, 112)
    
    # Forward pass
    logits, embedding = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Only embedding
    embedding_only = model(x, return_embedding=True)
    print(f"Embedding only shape: {embedding_only.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model size comparison
    print(f"Model size: {total_params / 1e6:.2f}M parameters")
