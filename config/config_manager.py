"""
Configuration manager for the RL-based taxi pricing system.
Handles loading, validation, and access to configuration parameters.
"""

import os
import yaml
import logging

logger = logging.getLogger("taxi-rl.config")

class ConfigManager:
    """Manages configuration for the RL-based taxi pricing system."""
    
    def __init__(self, config_path=None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path (str, optional): Path to config file. If None, uses default.
        """
        self.config_path = config_path
        self.config = None
    
    def create_default_config(self, save_path):
        """
        Create a default configuration with performance optimizations.
        
        Args:
            save_path (str): Path to save the default config.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Define default config
        config = {
            "env": {
                "api_key": "test_key",
                "max_rides": 10,
                "seed": 42
            },
            "model": {
                "context_dim": 6,
                "ride_feature_dim": 8,
                "max_rides": 10,
                "learning_rate": 3.0e-4,
                "buffer_size": 10000,
                "learning_starts": 500,
                "batch_size": 64,
                "tau": 1.0,
                "gamma": 0.99,
                "train_freq": 4,
                "gradient_steps": 1,
                "target_update_interval": 500,
                "exploration_fraction": 0.2,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
                "features_dim": 256,
                "max_grad_norm": 10,
                "seed": 42
            },
            "train": {
                "total_timesteps": 20000,
                "eval_freq": 2000,
                "n_eval_episodes": 5,
                "profile_freq": 5000,
                "checkpoint_freq": 5000
            },
            "performance": {
                "use_cache": True,
                "use_numba": True,
                "use_gpu": True,
                "parallel_processing": True,
                "vectorized_ops": True,
                "onnx_optimization": True
            }
        }
        
        # Save config
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Default configuration created at {save_path}")
        return config
    
    def load_config(self, config_path=None):
        """
        Load configuration from file.
        
        Args:
            config_path (str, optional): Path to config file. 
                                        If None, uses instance config_path.
        
        Returns:
            dict: Configuration dictionary.
        """
        path = config_path or self.config_path
        
        if not path:
            raise ValueError("No configuration path specified")
        
        try:
            with open(path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {path}")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_value(self, key_path, default=None):
        """
        Get a configuration value using a dot-separated path.
        
        Args:
            key_path (str): Dot-separated path to the config value (e.g., 'model.learning_rate')
            default: Default value to return if path not found
            
        Returns:
            The configuration value or default if not found.
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
            
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return default
                
        return value
    
    def update_config(self, key_path, value):
        """
        Update a configuration value using a dot-separated path.
        
        Args:
            key_path (str): Dot-separated path to the config value
            value: New value to set
        """
        if not self.config:
            raise ValueError("Configuration not loaded")
            
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the right level
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
    
    def save_config(self, path=None):
        """
        Save the current configuration to file.
        
        Args:
            path (str, optional): Path to save to. If None, uses instance config_path.
        """
        if not self.config:
            raise ValueError("No configuration to save")
            
        save_path = path or self.config_path
        
        if not save_path:
            raise ValueError("No save path specified")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        logger.info(f"Configuration saved to {save_path}")
