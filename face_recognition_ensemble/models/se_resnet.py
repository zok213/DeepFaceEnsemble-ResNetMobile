"""
SE-ResNet-50 implementation for face recognition.
Based on: "Squeeze-and-Excitation Networks" and face.evoLVe.PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


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


class SEBasicBlock(nn.Module):
    """SE-ResNet Basic Block."""
    
    expansion = 1
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None, reduction: int = 16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out, inplace=True)
        
        return out


class SEBottleneck(nn.Module):
    """SE-ResNet Bottleneck Block."""
    
    expansion = 4
    
    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None, reduction: int = 16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.se = SEBlock(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out, inplace=True)
        
        return out


class SEResNet(nn.Module):
    """SE-ResNet Architecture."""
    
    def __init__(self, block, layers: List[int], num_classes: int = 1000,
                 embedding_dim: int = 512, dropout: float = 0.5, 
                 reduction: int = 16):
        super(SEResNet, self).__init__()
        self.in_planes = 64
        self.reduction = reduction
        
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, 
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global pooling and embedding
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Linear(512 * block.expansion, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)
        
        # Classification head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, 
                           self.reduction))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, reduction=self.reduction))
        
        return nn.Sequential(*layers)
    
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
        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and embedding
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


def se_resnet50(num_classes: int = 1000, embedding_dim: int = 512, 
                dropout: float = 0.5, pretrained: bool = False) -> SEResNet:
    """SE-ResNet-50 model."""
    model = SEResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes,
                     embedding_dim=embedding_dim, dropout=dropout)
    
    if pretrained:
        # Load pretrained weights if available
        try:
            import torch.utils.model_zoo as model_zoo
            model_urls = {
                'se_resnet50': 'https://download.pytorch.org/models/se_resnet50-ce0d4300.pth'
            }
            pretrained_dict = model_zoo.load_url(model_urls['se_resnet50'])
            model_dict = model.state_dict()
            
            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
    
    return model


def se_resnet101(num_classes: int = 1000, embedding_dim: int = 512,
                 dropout: float = 0.5, pretrained: bool = False) -> SEResNet:
    """SE-ResNet-101 model."""
    model = SEResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes,
                     embedding_dim=embedding_dim, dropout=dropout)
    return model


# Test function
if __name__ == "__main__":
    # Test SE-ResNet-50
    model = se_resnet50(num_classes=8631, embedding_dim=512)
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
