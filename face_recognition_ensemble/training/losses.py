"""
Loss functions for face recognition.
Implements CosFace, ArcFace, Softmax, and other loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CosFaceLoss(nn.Module):
    """CosFace Loss Implementation.
    
    Reference: "CosFace: Large Margin Cosine Loss for Deep Face Recognition"
    """
    
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.4,
                 scale: float = 64.0, easy_margin: bool = False):
        super(CosFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # For numerical stability
        self.eps = 1e-8
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Feature embeddings [batch_size, embedding_dim]
            target: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        # Normalize input features and weights
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(input_norm, weight_norm)
        
        # Apply margin to target class
        phi = cosine - self.margin
        
        # Create one-hot encoding for target
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Compute cross entropy loss
        loss = F.cross_entropy(output, target)
        
        return loss


class ArcFaceLoss(nn.Module):
    """ArcFace Loss Implementation.
    
    Reference: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    """
    
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5,
                 scale: float = 64.0, easy_margin: bool = False):
        super(ArcFaceLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute values for numerical stability
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        self.eps = 1e-8
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Feature embeddings [batch_size, embedding_dim]
            target: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        # Normalize input features and weights
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(input_norm, weight_norm)
        
        # Compute sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute phi = cos(theta + margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding for target
        one_hot = torch.zeros(cosine.size()).to(input.device)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)
        
        # Apply margin only to target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        # Compute cross entropy loss
        loss = F.cross_entropy(output, target)
        
        return loss


class CircleLoss(nn.Module):
    """Circle Loss Implementation.
    
    Reference: "Circle Loss: A Unified Perspective of Pair-wise Similarity Optimization"
    """
    
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.25,
                 gamma: float = 256.0):
        super(CircleLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.gamma = gamma
        
        # Initialize weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Optimization parameters
        self.op = 1 + margin
        self.on = -margin
        self.delta_p = 1 - margin
        self.delta_n = margin
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Feature embeddings [batch_size, embedding_dim]
            target: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        # Normalize input features and weights
        input_norm = F.normalize(input, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = F.linear(input_norm, weight_norm)
        
        # Create target mask
        target_mask = torch.zeros_like(similarity)
        target_mask.scatter_(1, target.view(-1, 1), 1)
        
        # Separate positive and negative similarities
        sp = similarity * target_mask
        sn = similarity * (1 - target_mask)
        
        # Compute alpha and beta
        alpha_p = torch.clamp_min(self.op - sp.detach(), min=0.0)
        alpha_n = torch.clamp_min(sn.detach() - self.on, min=0.0)
        
        # Compute logits
        logit_p = -alpha_p * (sp - self.delta_p) * self.gamma
        logit_n = alpha_n * (sn - self.delta_n) * self.gamma
        
        # Combine logits
        logits = logit_p + logit_n
        
        # Compute loss
        loss = F.cross_entropy(logits, target)
        
        return loss


class CenterLoss(nn.Module):
    """Center Loss Implementation.
    
    Reference: "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """
    
    def __init__(self, embedding_dim: int, num_classes: int, alpha: float = 0.5):
        super(CenterLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.alpha = alpha
        
        # Initialize centers
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            input: Feature embeddings [batch_size, embedding_dim]
            target: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        batch_size = input.size(0)
        
        # Compute distances to centers
        centers_batch = self.centers.index_select(0, target.long())
        diff = input - centers_batch
        loss = torch.sum(torch.pow(diff, 2), dim=1) / 2.0
        
        return torch.mean(loss)


class TripletLoss(nn.Module):
    """Triplet Loss Implementation with hard mining."""
    
    def __init__(self, margin: float = 0.3, mining: str = 'hard'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
        
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            embeddings: Feature embeddings [batch_size, embedding_dim]
            labels: Ground truth labels [batch_size]
            
        Returns:
            Loss value
        """
        # Compute pairwise distances
        distances = self._pairwise_distances(embeddings)
        
        if self.mining == 'hard':
            return self._hard_triplet_loss(distances, labels)
        elif self.mining == 'semi_hard':
            return self._semi_hard_triplet_loss(distances, labels)
        else:
            return self._batch_all_triplet_loss(distances, labels)
    
    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared distances."""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)
        
        return distances
    
    def _hard_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Hard triplet loss with hard negative mining."""
        batch_size = labels.size(0)
        
        # Create masks for positive and negative pairs
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        
        # Hard positive mining
        anchor_positive_dist = distances.unsqueeze(2)
        anchor_positive_dist = anchor_positive_dist * mask_anchor_positive.float()
        hardest_positive_dist = torch.max(anchor_positive_dist, dim=2)[0]
        
        # Hard negative mining
        max_anchor_negative_dist = torch.max(distances, dim=1, keepdim=True)[0]
        anchor_negative_dist = distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist = torch.min(anchor_negative_dist, dim=1)[0]
        
        # Triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return torch.mean(triplet_loss)
    
    def _semi_hard_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Semi-hard triplet loss."""
        # Implementation for semi-hard mining
        # This is a simplified version
        return self._hard_triplet_loss(distances, labels)
    
    def _batch_all_triplet_loss(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Batch all triplet loss."""
        # Implementation for batch all mining
        # This is a simplified version
        return self._hard_triplet_loss(distances, labels)
    
    def _get_anchor_positive_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get mask for anchor-positive pairs."""
        indices_equal = torch.eye(labels.size(0)).bool().to(labels.device)
        indices_not_equal = ~indices_equal
        
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        return labels_equal & indices_not_equal
    
    def _get_anchor_negative_triplet_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """Get mask for anchor-negative pairs."""
        return labels.unsqueeze(0) != labels.unsqueeze(1)


class CombinedLoss(nn.Module):
    """Combined loss function for face recognition."""
    
    def __init__(self, losses: dict, weights: Optional[dict] = None):
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)
        
        if weights is None:
            self.weights = {name: 1.0 for name in losses.keys()}
        else:
            self.weights = weights
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> dict:
        """Forward pass.
        
        Args:
            input: Feature embeddings [batch_size, embedding_dim]
            target: Ground truth labels [batch_size]
            
        Returns:
            Dictionary of loss values
        """
        total_loss = 0.0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(input, target)
            weighted_loss = self.weights.get(name, 1.0) * loss_value
            
            loss_dict[name] = loss_value
            total_loss += weighted_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict


# Factory function to create loss functions
def create_loss_function(loss_type: str, embedding_dim: int, num_classes: int, **kwargs):
    """Create loss function based on type."""
    if loss_type.lower() == 'cosface':
        return CosFaceLoss(embedding_dim, num_classes, **kwargs)
    elif loss_type.lower() == 'arcface':
        return ArcFaceLoss(embedding_dim, num_classes, **kwargs)
    elif loss_type.lower() == 'circle':
        return CircleLoss(embedding_dim, num_classes, **kwargs)
    elif loss_type.lower() == 'center':
        return CenterLoss(embedding_dim, num_classes, **kwargs)
    elif loss_type.lower() == 'triplet':
        return TripletLoss(**kwargs)
    elif loss_type.lower() == 'softmax':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Test function
if __name__ == "__main__":
    # Test loss functions
    batch_size = 32
    embedding_dim = 512
    num_classes = 1000
    
    # Create dummy data
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test CosFace loss
    cosface_loss = CosFaceLoss(embedding_dim, num_classes)
    loss = cosface_loss(embeddings, labels)
    print(f"CosFace loss: {loss.item():.4f}")
    
    # Test ArcFace loss
    arcface_loss = ArcFaceLoss(embedding_dim, num_classes)
    loss = arcface_loss(embeddings, labels)
    print(f"ArcFace loss: {loss.item():.4f}")
    
    # Test Circle loss
    circle_loss = CircleLoss(embedding_dim, num_classes)
    loss = circle_loss(embeddings, labels)
    print(f"Circle loss: {loss.item():.4f}")
    
    # Test Center loss
    center_loss = CenterLoss(embedding_dim, num_classes)
    loss = center_loss(embeddings, labels)
    print(f"Center loss: {loss.item():.4f}")
    
    # Test Triplet loss
    triplet_loss = TripletLoss()
    loss = triplet_loss(embeddings, labels)
    print(f"Triplet loss: {loss.item():.4f}")
    
    # Test Combined loss
    combined_loss = CombinedLoss({
        'cosface': CosFaceLoss(embedding_dim, num_classes),
        'center': CenterLoss(embedding_dim, num_classes)
    }, weights={'cosface': 1.0, 'center': 0.1})
    
    loss_dict = combined_loss(embeddings, labels)
    print(f"Combined loss: {loss_dict}")
