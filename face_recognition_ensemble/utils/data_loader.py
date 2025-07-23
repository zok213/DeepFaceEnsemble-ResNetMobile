"""
Data loading utilities for face recognition ensemble.
"""

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image as Image
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd


class VGGFace2Dataset(Dataset):
    """VGGFace2 dataset loader."""
    
    def __init__(self, root_dir: str, annotation_file: str, 
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None):
        """Initialize VGGFace2 dataset.
        
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to annotation file
            transform: Image transformations
            target_transform: Target transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Load annotations
        self.annotations = self._load_annotations(annotation_file)
        
        # Create label mapping
        self.label_to_idx = self._create_label_mapping()
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Filter annotations with valid labels
        self.annotations = self.annotations[
            self.annotations['label'].isin(self.label_to_idx.keys())
        ].reset_index(drop=True)
        
        print(f"Loaded {len(self.annotations)} samples from {len(self.label_to_idx)} classes")
    
    def _load_annotations(self, annotation_file: str) -> pd.DataFrame:
        """Load annotations from file."""
        if annotation_file.endswith('.csv'):
            return pd.read_csv(annotation_file)
        elif annotation_file.endswith('.txt'):
            # Assume format: image_path label
            data = []
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label = parts[1]
                        data.append({'image_path': image_path, 'label': label})
            return pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported annotation file format: {annotation_file}")
    
    def _create_label_mapping(self) -> Dict[str, int]:
        """Create mapping from label names to indices."""
        unique_labels = sorted(self.annotations['label'].unique())
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.root_dir, row['image_path'])
        label = row['label']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (112, 112), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert label to index
        label_idx = self.label_to_idx[label]
        
        if self.target_transform:
            label_idx = self.target_transform(label_idx)
        
        return image, label_idx
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        label_counts = self.annotations['label'].value_counts()
        total_samples = len(self.annotations)
        
        weights = []
        for label in sorted(self.label_to_idx.keys()):
            count = label_counts.get(label, 1)
            weight = total_samples / (len(self.label_to_idx) * count)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class IJBCDataset(Dataset):
    """IJB-C dataset loader for evaluation."""
    
    def __init__(self, root_dir: str, protocol_file: str,
                 transform: Optional[transforms.Compose] = None):
        """Initialize IJB-C dataset.
        
        Args:
            root_dir: Root directory containing images
            protocol_file: Path to protocol file
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load protocol
        self.protocol = self._load_protocol(protocol_file)
        
        print(f"Loaded {len(self.protocol)} samples from IJB-C")
    
    def _load_protocol(self, protocol_file: str) -> pd.DataFrame:
        """Load protocol from file."""
        return pd.read_csv(protocol_file)
    
    def __len__(self) -> int:
        return len(self.protocol)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item by index."""
        row = self.protocol.iloc[idx]
        image_path = os.path.join(self.root_dir, row['FILENAME'])
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (112, 112), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return image and metadata
        metadata = {
            'subject_id': row['SUBJECT_ID'],
            'template_id': row['TEMPLATE_ID'],
            'filename': row['FILENAME']
        }
        
        return image, metadata


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Get training transforms."""
    data_config = config.get('DATA', {})
    augmentation = data_config.get('augmentation', {})
    
    transform_list = []
    
    # Resize
    image_size = data_config.get('image_size', [112, 112])
    transform_list.append(transforms.Resize(image_size))
    
    # Data augmentation
    if augmentation.get('horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    if augmentation.get('rotation', 0) > 0:
        transform_list.append(transforms.RandomRotation(augmentation['rotation']))
    
    if augmentation.get('color_jitter', 0) > 0:
        transform_list.append(transforms.ColorJitter(
            brightness=augmentation['color_jitter'],
            contrast=augmentation['color_jitter'],
            saturation=augmentation['color_jitter'],
            hue=augmentation['color_jitter'] / 4
        ))
    
    if augmentation.get('random_crop', False):
        transform_list.append(transforms.RandomCrop(image_size, padding=4))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if augmentation.get('normalize', True):
        mean = data_config.get('mean', [0.485, 0.456, 0.406])
        std = data_config.get('std', [0.229, 0.224, 0.225])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """Get validation transforms."""
    data_config = config.get('DATA', {})
    
    transform_list = []
    
    # Resize
    image_size = data_config.get('image_size', [112, 112])
    transform_list.append(transforms.Resize(image_size))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    mean = data_config.get('mean', [0.485, 0.456, 0.406])
    std = data_config.get('std', [0.229, 0.224, 0.225])
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def create_data_loaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    # Get paths
    train_config = config.get('TRAIN', {})
    data_config = config.get('DATA', {})
    
    dataset_root = train_config.get('dataset_root', './data/VGGFace2')
    train_list = train_config.get('train_list', './data/VGGFace2/train_list.txt')
    val_list = train_config.get('val_list', './data/VGGFace2/val_list.txt')
    
    # Create transforms
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    
    # Create datasets
    train_dataset = VGGFace2Dataset(
        root_dir=dataset_root,
        annotation_file=train_list,
        transform=train_transform
    )
    
    val_dataset = VGGFace2Dataset(
        root_dir=dataset_root,
        annotation_file=val_list,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get('batch_size', 64),
        shuffle=True,
        num_workers=data_config.get('num_workers', 8),
        pin_memory=data_config.get('pin_memory', True),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get('batch_size', 64),
        shuffle=False,
        num_workers=data_config.get('num_workers', 8),
        pin_memory=data_config.get('pin_memory', True),
        drop_last=False
    )
    
    return train_loader, val_loader


def create_ijbc_loader(config: Dict[str, Any]) -> DataLoader:
    """Create IJB-C data loader for evaluation."""
    
    eval_config = config.get('EVAL', {})
    
    ijbc_root = eval_config.get('ijb_c_root', './data/IJB-C')
    protocol_file = os.path.join(ijbc_root, 'protocols', 'ijbc_1N_probe_mixed.csv')
    
    # Create transforms
    val_transform = get_val_transforms(config)
    
    # Create dataset
    ijbc_dataset = IJBCDataset(
        root_dir=ijbc_root,
        protocol_file=protocol_file,
        transform=val_transform
    )
    
    # Create data loader
    ijbc_loader = DataLoader(
        ijbc_dataset,
        batch_size=eval_config.get('eval_batch_size', 256),
        shuffle=False,
        num_workers=config.get('DATA', {}).get('num_workers', 8),
        pin_memory=config.get('DATA', {}).get('pin_memory', True),
        drop_last=False
    )
    
    return ijbc_loader


class BalancedBatchSampler:
    """Balanced batch sampler for face recognition."""
    
    def __init__(self, dataset: Dataset, batch_size: int, samples_per_class: int = 2):
        """Initialize balanced batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            samples_per_class: Number of samples per class in each batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        
        # Group samples by class
        self.class_to_indices = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        self.num_classes = len(self.class_to_indices)
        self.classes_per_batch = batch_size // samples_per_class
        
    def __iter__(self):
        """Iterate over batches."""
        while True:
            # Select random classes
            selected_classes = np.random.choice(
                list(self.class_to_indices.keys()),
                size=self.classes_per_batch,
                replace=False
            )
            
            batch_indices = []
            for cls in selected_classes:
                # Select random samples from this class
                class_indices = np.random.choice(
                    self.class_to_indices[cls],
                    size=self.samples_per_class,
                    replace=len(self.class_to_indices[cls]) < self.samples_per_class
                )
                batch_indices.extend(class_indices)
            
            yield batch_indices
    
    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.dataset) // self.batch_size


# Test function
if __name__ == "__main__":
    # Test data loading
    config = {
        'TRAIN': {
            'dataset_root': './data/VGGFace2',
            'train_list': './data/VGGFace2/train_list.txt',
            'val_list': './data/VGGFace2/val_list.txt',
            'batch_size': 32
        },
        'DATA': {
            'image_size': [112, 112],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'augmentation': {
                'horizontal_flip': True,
                'rotation': 10,
                'color_jitter': 0.2,
                'random_crop': True,
                'normalize': True
            },
            'num_workers': 4,
            'pin_memory': True
        }
    }
    
    # Create dummy annotation files for testing
    os.makedirs('./data/VGGFace2', exist_ok=True)
    
    # Create dummy train list
    with open('./data/VGGFace2/train_list.txt', 'w') as f:
        for i in range(100):
            f.write(f"person_{i:03d}/img_{i:03d}.jpg person_{i:03d}\n")
    
    # Create dummy val list
    with open('./data/VGGFace2/val_list.txt', 'w') as f:
        for i in range(20):
            f.write(f"person_{i:03d}/img_{i:03d}_val.jpg person_{i:03d}\n")
    
    try:
        # Test data loader creation
        train_loader, val_loader = create_data_loaders(config)
        print(f"Created train loader with {len(train_loader)} batches")
        print(f"Created val loader with {len(val_loader)} batches")
        
        # Test transforms
        train_transforms = get_train_transforms(config)
        val_transforms = get_val_transforms(config)
        print(f"Train transforms: {train_transforms}")
        print(f"Val transforms: {val_transforms}")
        
    except Exception as e:
        print(f"Error testing data loader: {e}")
        print("This is expected since we don't have actual image data")
