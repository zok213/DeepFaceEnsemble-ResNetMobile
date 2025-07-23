"""
Ensemble methods for face recognition.
Implements feature averaging, weighted voting, and advanced ensemble techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score
import logging

from ..models.se_resnet import se_resnet50
from ..models.mobilefacenet import mobilefacenet


class EnsembleModel(nn.Module):
    """Ensemble model combining multiple face recognition models."""
    
    def __init__(self, models: List[nn.Module], ensemble_method: str = 'weighted_average',
                 weights: Optional[List[float]] = None, temperature: float = 1.0,
                 num_classes: int = 1000, embedding_dim: int = 512):
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.temperature = temperature
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Initialize weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            assert len(weights) == len(models), "Number of weights must match number of models"
            self.weights = weights
        
        # Convert to tensor for GPU computation
        self.register_buffer('weight_tensor', torch.tensor(self.weights))
        
        # Optional learned ensemble weights
        if ensemble_method == 'learned_weights':
            self.learned_weights = nn.Parameter(torch.ones(len(models)))
            self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False, 
                return_individual: bool = False):
        """Forward pass through ensemble."""
        embeddings = []
        logits = []
        
        # Get outputs from each model
        for model in self.models:
            if return_embedding:
                emb = model(x, return_embedding=True)
                embeddings.append(emb)
            else:
                logit, emb = model(x)
                logits.append(logit)
                embeddings.append(emb)
        
        # Stack tensors
        if embeddings:
            embeddings = torch.stack(embeddings, dim=0)  # [num_models, batch_size, embedding_dim]
        if logits:
            logits = torch.stack(logits, dim=0)  # [num_models, batch_size, num_classes]
        
        # Apply ensemble method
        if self.ensemble_method == 'average':
            ensemble_embedding = torch.mean(embeddings, dim=0)
            ensemble_logits = torch.mean(logits, dim=0) if logits.size(0) > 0 else None
            
        elif self.ensemble_method == 'weighted_average':
            weights = self.weight_tensor.view(-1, 1, 1)
            ensemble_embedding = torch.sum(embeddings * weights, dim=0)
            if logits.size(0) > 0:
                ensemble_logits = torch.sum(logits * weights, dim=0)
            else:
                ensemble_logits = None
                
        elif self.ensemble_method == 'learned_weights':
            weights = self.softmax(self.learned_weights).view(-1, 1, 1)
            ensemble_embedding = torch.sum(embeddings * weights, dim=0)
            if logits.size(0) > 0:
                ensemble_logits = torch.sum(logits * weights, dim=0)
            else:
                ensemble_logits = None
                
        elif self.ensemble_method == 'voting':
            # For voting, we need logits
            if logits.size(0) == 0:
                raise ValueError("Voting requires logits, but return_embedding=True")
            
            # Softmax on individual predictions
            probs = F.softmax(logits / self.temperature, dim=2)
            ensemble_probs = torch.mean(probs, dim=0)
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            ensemble_embedding = torch.mean(embeddings, dim=0)
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Normalize embeddings
        ensemble_embedding = F.normalize(ensemble_embedding, p=2, dim=1)
        
        if return_individual:
            return (ensemble_logits, ensemble_embedding, 
                   logits.unbind(0), embeddings.unbind(0))
        
        if return_embedding:
            return ensemble_embedding
        
        return ensemble_logits, ensemble_embedding


class AdaptiveEnsemble(nn.Module):
    """Adaptive ensemble that learns to weight models based on input."""
    
    def __init__(self, models: List[nn.Module], embedding_dim: int = 512,
                 num_classes: int = 1000, hidden_dim: int = 256):
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.embedding_dim = embedding_dim
        
        # Attention mechanism to compute adaptive weights
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * self.num_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_models),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """Forward pass with adaptive weighting."""
        embeddings = []
        logits = []
        
        # Get outputs from each model
        for model in self.models:
            if return_embedding:
                emb = model(x, return_embedding=True)
                embeddings.append(emb)
            else:
                logit, emb = model(x)
                logits.append(logit)
                embeddings.append(emb)
        
        # Stack embeddings
        embeddings = torch.stack(embeddings, dim=0)  # [num_models, batch_size, embedding_dim]
        embeddings = embeddings.transpose(0, 1)  # [batch_size, num_models, embedding_dim]
        
        # Compute attention weights
        concat_embeddings = embeddings.view(embeddings.size(0), -1)
        attention_weights = self.attention(concat_embeddings)  # [batch_size, num_models]
        
        # Apply attention weights
        attention_weights = attention_weights.unsqueeze(2)  # [batch_size, num_models, 1]
        weighted_embeddings = embeddings * attention_weights
        ensemble_embedding = torch.sum(weighted_embeddings, dim=1)  # [batch_size, embedding_dim]
        
        # Final projection
        ensemble_embedding = self.final_projection(ensemble_embedding)
        ensemble_embedding = F.normalize(ensemble_embedding, p=2, dim=1)
        
        if return_embedding:
            return ensemble_embedding
        
        # Classification
        ensemble_logits = self.classifier(ensemble_embedding)
        
        return ensemble_logits, ensemble_embedding


class EnsembleTrainer:
    """Trainer for ensemble models with various strategies."""
    
    def __init__(self, models: List[nn.Module], device: torch.device = torch.device('cpu')):
        self.models = models
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Move models to device
        for model in self.models:
            model.to(device)
    
    def train_individual_models(self, train_loader, val_loader, epochs: int = 10,
                               lr: float = 0.001, weight_decay: float = 1e-4):
        """Train individual models separately."""
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i+1}/{len(self.models)}")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    logits, _ = model(data)
                    loss = criterion(logits, target)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                if val_loader is not None:
                    val_acc = self.evaluate_model(model, val_loader)
                    self.logger.info(f"Model {i+1}, Epoch {epoch+1}, "
                                   f"Train Loss: {train_loss/len(train_loader):.4f}, "
                                   f"Val Acc: {val_acc:.4f}")
    
    def evaluate_model(self, model: nn.Module, data_loader) -> float:
        """Evaluate a single model."""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits, _ = model(data)
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total
    
    def find_optimal_weights(self, val_loader, method: str = 'grid_search') -> List[float]:
        """Find optimal ensemble weights using validation set."""
        if method == 'grid_search':
            return self._grid_search_weights(val_loader)
        elif method == 'bayesian_optimization':
            return self._bayesian_optimization_weights(val_loader)
        else:
            raise ValueError(f"Unknown weight optimization method: {method}")
    
    def _grid_search_weights(self, val_loader, num_steps: int = 11) -> List[float]:
        """Grid search for optimal weights."""
        best_weights = None
        best_acc = 0.0
        
        # Generate weight combinations
        if len(self.models) == 2:
            for w1 in np.linspace(0.1, 0.9, num_steps):
                w2 = 1.0 - w1
                weights = [w1, w2]
                acc = self._evaluate_ensemble_weights(val_loader, weights)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights
        
        elif len(self.models) == 3:
            for w1 in np.linspace(0.1, 0.8, num_steps):
                for w2 in np.linspace(0.1, 0.9 - w1, num_steps):
                    w3 = 1.0 - w1 - w2
                    if w3 > 0.1:
                        weights = [w1, w2, w3]
                        acc = self._evaluate_ensemble_weights(val_loader, weights)
                        
                        if acc > best_acc:
                            best_acc = acc
                            best_weights = weights
        
        else:
            # For more models, use random sampling
            best_weights = [1.0 / len(self.models)] * len(self.models)
            for _ in range(1000):
                weights = np.random.dirichlet(np.ones(len(self.models)))
                weights = weights.tolist()
                acc = self._evaluate_ensemble_weights(val_loader, weights)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = weights
        
        self.logger.info(f"Best ensemble weights: {best_weights}, Accuracy: {best_acc:.4f}")
        return best_weights
    
    def _evaluate_ensemble_weights(self, val_loader, weights: List[float]) -> float:
        """Evaluate ensemble with given weights."""
        # Set models to evaluation mode
        for model in self.models:
            model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get predictions from all models
                logits_list = []
                for model in self.models:
                    logits, _ = model(data)
                    logits_list.append(logits)
                
                # Weighted ensemble
                logits_stack = torch.stack(logits_list, dim=0)
                weights_tensor = torch.tensor(weights, device=self.device).view(-1, 1, 1)
                ensemble_logits = torch.sum(logits_stack * weights_tensor, dim=0)
                
                pred = ensemble_logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return correct / total


def create_ensemble_model(model_configs: List[Dict], ensemble_method: str = 'weighted_average',
                         weights: Optional[List[float]] = None, num_classes: int = 8631,
                         embedding_dim: int = 512) -> EnsembleModel:
    """Create ensemble model from configuration."""
    models = []
    
    for config in model_configs:
        if config['type'] == 'se_resnet50':
            model = se_resnet50(num_classes=num_classes, embedding_dim=embedding_dim,
                               dropout=config.get('dropout', 0.5))
        elif config['type'] == 'mobilefacenet':
            model = mobilefacenet(num_classes=num_classes, embedding_dim=embedding_dim,
                                 dropout=config.get('dropout', 0.5))
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
        
        # Load pretrained weights if specified
        if 'pretrained_path' in config and config['pretrained_path']:
            try:
                model.load_state_dict(torch.load(config['pretrained_path']))
                print(f"Loaded pretrained weights for {config['type']}")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights: {e}")
        
        models.append(model)
    
    return EnsembleModel(models, ensemble_method=ensemble_method, weights=weights,
                        num_classes=num_classes, embedding_dim=embedding_dim)


# Test function
if __name__ == "__main__":
    # Test ensemble
    model1 = se_resnet50(num_classes=1000, embedding_dim=512)
    model2 = mobilefacenet(num_classes=1000, embedding_dim=512)
    
    ensemble = EnsembleModel([model1, model2], ensemble_method='weighted_average',
                            weights=[0.6, 0.4])
    
    x = torch.randn(4, 3, 112, 112)
    
    # Test forward pass
    logits, embedding = ensemble(x)
    print(f"Ensemble logits shape: {logits.shape}")
    print(f"Ensemble embedding shape: {embedding.shape}")
    
    # Test embedding only
    embedding_only = ensemble(x, return_embedding=True)
    print(f"Ensemble embedding only shape: {embedding_only.shape}")
    
    # Test individual outputs
    logits, embedding, ind_logits, ind_embeddings = ensemble(x, return_individual=True)
    print(f"Individual logits: {len(ind_logits)}")
    print(f"Individual embeddings: {len(ind_embeddings)}")
    
    # Test adaptive ensemble
    adaptive = AdaptiveEnsemble([model1, model2], embedding_dim=512, num_classes=1000)
    adaptive_logits, adaptive_embedding = adaptive(x)
    print(f"Adaptive ensemble logits shape: {adaptive_logits.shape}")
    print(f"Adaptive ensemble embedding shape: {adaptive_embedding.shape}")
