"""
Main training script for face recognition ensemble.
"""

import os
import sys
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import get_config, cfg
from models.se_resnet import se_resnet50
from models.mobilefacenet import mobilefacenet
from ensemble.ensemble_model import EnsembleModel, EnsembleTrainer
from training.losses import create_loss_function
from utils.data_loader import create_data_loaders
from utils.metrics import calculate_metrics
from utils.logger import setup_logger
from utils.checkpoint import save_checkpoint, load_checkpoint


class EnsembleTrainingPipeline:
    """Complete training pipeline for ensemble face recognition."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        if config_path:
            self.cfg = get_config(config_path)
        else:
            self.cfg = cfg
        
        # Setup logging
        self.logger = setup_logger(self.cfg.get('ENV.log_dir', './logs'))
        
        # Setup device
        self.device = torch.device(self.cfg.get('HARDWARE.device', 'cuda') 
                                 if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Setup directories
        self.cfg.create_directories()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(self.cfg.get('ENV.tensorboard_dir', './tensorboard'))
        
        # Initialize models
        self.models = self._create_models()
        self.ensemble_model = self._create_ensemble_model()
        
        # Initialize loss function
        self.loss_fn = self._create_loss_function()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.best_acc = 0.0
        self.global_step = 0
        
    def _create_models(self) -> List[nn.Module]:
        """Create individual models."""
        models = []
        model_configs = self.cfg.get('MODEL')
        
        # SE-ResNet-50
        se_resnet_config = model_configs.get('se_resnet', {})
        se_resnet_model = se_resnet50(
            num_classes=self.cfg.get('TRAIN.num_classes', 8631),
            embedding_dim=se_resnet_config.get('embedding_dim', 512),
            dropout=se_resnet_config.get('dropout', 0.5)
        )
        models.append(se_resnet_model)
        
        # MobileFaceNet
        mobile_config = model_configs.get('mobile_facenet', {})
        mobile_model = mobilefacenet(
            num_classes=self.cfg.get('TRAIN.num_classes', 8631),
            embedding_dim=mobile_config.get('embedding_dim', 512),
            dropout=mobile_config.get('dropout', 0.5)
        )
        models.append(mobile_model)
        
        # Move models to device
        for model in models:
            model.to(self.device)
        
        # Load pretrained weights if available
        self._load_pretrained_weights(models)
        
        return models
    
    def _create_ensemble_model(self) -> EnsembleModel:
        """Create ensemble model."""
        ensemble_config = self.cfg.get('MODEL.ensemble', {})
        
        ensemble_model = EnsembleModel(
            models=self.models,
            ensemble_method=ensemble_config.get('method', 'weighted_average'),
            weights=ensemble_config.get('weights', [0.6, 0.4]),
            temperature=ensemble_config.get('temperature', 1.0),
            num_classes=self.cfg.get('TRAIN.num_classes', 8631),
            embedding_dim=512
        )
        
        ensemble_model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1 and self.cfg.get('TRAIN.use_parallel', False):
            ensemble_model = nn.DataParallel(ensemble_model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        
        return ensemble_model
    
    def _create_loss_function(self):
        """Create loss function."""
        loss_type = self.cfg.get('TRAIN.loss_type', 'CosFace')
        
        return create_loss_function(
            loss_type=loss_type,
            embedding_dim=512,
            num_classes=self.cfg.get('TRAIN.num_classes', 8631),
            margin=self.cfg.get('TRAIN.margin', 0.4),
            scale=self.cfg.get('TRAIN.scale', 64)
        )
    
    def _create_optimizer(self):
        """Create optimizer."""
        optimizer_type = self.cfg.get('TRAIN.optimizer', 'SGD')
        
        if optimizer_type == 'SGD':
            return optim.SGD(
                self.ensemble_model.parameters(),
                lr=self.cfg.get('TRAIN.learning_rate', 0.001),
                momentum=self.cfg.get('TRAIN.momentum', 0.9),
                weight_decay=self.cfg.get('TRAIN.weight_decay', 0.0005)
            )
        elif optimizer_type == 'Adam':
            return optim.Adam(
                self.ensemble_model.parameters(),
                lr=self.cfg.get('TRAIN.learning_rate', 0.001),
                weight_decay=self.cfg.get('TRAIN.weight_decay', 0.0005)
            )
        elif optimizer_type == 'AdamW':
            return optim.AdamW(
                self.ensemble_model.parameters(),
                lr=self.cfg.get('TRAIN.learning_rate', 0.001),
                weight_decay=self.cfg.get('TRAIN.weight_decay', 0.0005)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.cfg.get('TRAIN.lr_scheduler', 'StepLR')
        
        if scheduler_type == 'StepLR':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.get('TRAIN.step_size', 20),
                gamma=self.cfg.get('TRAIN.gamma', 0.1)
            )
        elif scheduler_type == 'CosineAnnealingLR':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.get('TRAIN.epochs', 100)
            )
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def _create_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders."""
        return create_data_loaders(self.cfg)
    
    def _load_pretrained_weights(self, models: List[nn.Module]):
        """Load pretrained weights for models."""
        model_configs = self.cfg.get('MODEL')
        
        # Load SE-ResNet-50 weights
        se_resnet_path = model_configs.get('se_resnet', {}).get('pretrained_path')
        if se_resnet_path and os.path.exists(se_resnet_path):
            try:
                models[0].load_state_dict(torch.load(se_resnet_path))
                self.logger.info(f"Loaded SE-ResNet-50 weights from {se_resnet_path}")
            except Exception as e:
                self.logger.warning(f"Could not load SE-ResNet-50 weights: {e}")
        
        # Load MobileFaceNet weights
        mobile_path = model_configs.get('mobile_facenet', {}).get('pretrained_path')
        if mobile_path and os.path.exists(mobile_path):
            try:
                models[1].load_state_dict(torch.load(mobile_path))
                self.logger.info(f"Loaded MobileFaceNet weights from {mobile_path}")
            except Exception as e:
                self.logger.warning(f"Could not load MobileFaceNet weights: {e}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.ensemble_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.loss_fn, 'weight'):
                # For metric learning losses (CosFace, ArcFace)
                logits, embeddings = self.ensemble_model(data)
                loss = self.loss_fn(embeddings, target)
            else:
                # For standard losses
                logits, embeddings = self.ensemble_model(data)
                loss = self.loss_fn(logits, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
            
            # Log to tensorboard
            if batch_idx % self.cfg.get('ENV.print_freq', 100) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy', 100. * correct / total, self.global_step)
                self.writer.add_scalar('Train/Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.ensemble_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                if hasattr(self.loss_fn, 'weight'):
                    logits, embeddings = self.ensemble_model(data)
                    loss = self.loss_fn(embeddings, target)
                else:
                    logits, embeddings = self.ensemble_model(data)
                    loss = self.loss_fn(logits, target)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting ensemble training...")
        self.logger.info(f"Training for {self.cfg.get('TRAIN.epochs', 100)} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.cfg.get('TRAIN.epochs', 100)):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['accuracy'])
            else:
                self.scheduler.step()
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%"
            )
            
            # Tensorboard logging
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_metrics['accuracy'], epoch)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['accuracy']
            
            if (epoch + 1) % self.cfg.get('TRAIN.save_interval', 10) == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.ensemble_model.state_dict(),
                    'best_acc': self.best_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'config': dict(self.cfg.config)
                }, is_best, self.cfg.get('TRAIN.checkpoint_dir', './checkpoints'))
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation accuracy: {self.best_acc:.2f}%")
        
        # Close tensorboard writer
        self.writer.close()
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path)
        
        self.ensemble_model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.current_epoch = checkpoint['epoch']
        self.best_acc = checkpoint['best_acc']
        
        self.logger.info(f"Resumed training from epoch {self.current_epoch}")
        self.logger.info(f"Best accuracy so far: {self.best_acc:.2f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train face recognition ensemble")
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device to use')
    
    args = parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    
    # Create training pipeline
    pipeline = EnsembleTrainingPipeline(args.config)
    
    # Resume training if specified
    if args.resume:
        pipeline.resume_training(args.resume)
    
    # Start training
    pipeline.train()


if __name__ == "__main__":
    main()
