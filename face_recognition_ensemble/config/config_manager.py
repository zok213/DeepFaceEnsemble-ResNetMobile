import yaml
import os
from easydict import EasyDict
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the face recognition ensemble project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> EasyDict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return EasyDict(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'TRAIN.epochs')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = EasyDict()
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        for key, value in updates.items():
            self.set(key, value)
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path.
        """
        if path is None:
            path = self.config_path
        
        with open(path, 'w') as f:
            yaml.dump(dict(self.config), f, default_flow_style=False)
    
    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            self.get('TRAIN.checkpoint_dir', './checkpoints'),
            self.get('ENV.output_dir', './outputs'),
            self.get('ENV.log_dir', './logs'),
            self.get('ENV.tensorboard_dir', './tensorboard'),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        self._print_dict(self.config)
        print("=" * 50)
    
    def _print_dict(self, d: dict, indent: int = 0) -> None:
        """Recursively print dictionary."""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


# Global configuration instance
cfg = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration instance
    """
    if config_path is not None:
        return Config(config_path)
    return cfg


def update_config(updates: Dict[str, Any]) -> None:
    """Update global configuration.
    
    Args:
        updates: Dictionary of updates
    """
    cfg.update(updates)


# Convenience functions for common configurations
def get_train_config() -> EasyDict:
    """Get training configuration."""
    return cfg.get('TRAIN')


def get_model_config() -> EasyDict:
    """Get model configuration."""
    return cfg.get('MODEL')


def get_data_config() -> EasyDict:
    """Get data configuration."""
    return cfg.get('DATA')


def get_eval_config() -> EasyDict:
    """Get evaluation configuration."""
    return cfg.get('EVAL')


def get_env_config() -> EasyDict:
    """Get environment configuration."""
    return cfg.get('ENV')


def get_hardware_config() -> EasyDict:
    """Get hardware configuration."""
    return cfg.get('HARDWARE')
