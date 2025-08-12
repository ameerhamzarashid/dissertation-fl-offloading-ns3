"""
File Manager - Handles model saving, loading, and data persistence
Provides centralized file I/O operations for the FL system
"""

import os
import pickle
import json
import torch
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional, List
import shutil

from utils.logger import setup_logger


class FileManager:
    """
    Centralized file management for federated learning system.
    """
    
    def __init__(self, base_directory: str = 'data'):
        """
        Initialize file manager with base directory.
        
        Args:
            base_directory: Base directory for all file operations
        """
        self.base_directory = base_directory
        self.logger = setup_logger("FileManager")
        
        # Create directory structure
        self.directories = {
            'models': os.path.join(base_directory, 'models'),
            'checkpoints': os.path.join(base_directory, 'checkpoints'),
            'raw_data': os.path.join(base_directory, 'raw'),
            'processed_data': os.path.join(base_directory, 'processed'),
            'results': os.path.join(base_directory, 'results'),
            'configs': os.path.join(base_directory, 'configs'),
            'logs': os.path.join(base_directory, 'logs')
        }
        
        # Create directories if they don't exist
        for directory in self.directories.values():
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"File manager initialized with base directory: {base_directory}")
    
    def save_model(self, model: torch.nn.Module, filename: str, 
                   additional_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save a PyTorch model with additional metadata.
        
        Args:
            model: PyTorch model to save
            filename: Filename for the saved model
            additional_info: Additional information to save with the model
            
        Returns:
            Full path to the saved model file
        """
        filepath = os.path.join(self.directories['models'], filename)
        
        # Prepare save data
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_architecture': str(model),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Add additional info if provided
        if additional_info:
            save_data.update(additional_info)
        
        # Save the model
        torch.save(save_data, filepath)
        self.logger.info(f"Model saved to: {filepath}")
        
        return filepath
    
    def load_model(self, model: torch.nn.Module, filename: str) -> Dict[str, Any]:
        """
        Load a PyTorch model and return metadata.
        
        Args:
            model: PyTorch model to load state into
            filename: Filename of the saved model
            
        Returns:
            Dictionary containing model metadata
        """
        filepath = os.path.join(self.directories['models'], filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the saved data
        save_data = torch.load(filepath, map_location='cpu')
        
        # Load model state
        model.load_state_dict(save_data['model_state_dict'])
        
        # Extract metadata
        metadata = {k: v for k, v in save_data.items() if k != 'model_state_dict'}
        
        self.logger.info(f"Model loaded from: {filepath}")
        return metadata
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any], 
                       experiment_name: str, round_num: int) -> str:
        """
        Save a training checkpoint.
        
        Args:
            checkpoint_data: Dictionary containing checkpoint data
            experiment_name: Name of the experiment
            round_num: Current round number
            
        Returns:
            Full path to the saved checkpoint file
        """
        filename = f"{experiment_name}_checkpoint_round_{round_num}.pt"
        filepath = os.path.join(self.directories['checkpoints'], filename)
        
        # Add timestamp to checkpoint data
        checkpoint_data['checkpoint_timestamp'] = datetime.now().isoformat()
        checkpoint_data['round_number'] = round_num
        checkpoint_data['experiment_name'] = experiment_name
        
        torch.save(checkpoint_data, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
        
        return filepath
    
    def load_checkpoint(self, experiment_name: str, round_num: int) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            experiment_name: Name of the experiment
            round_num: Round number to load
            
        Returns:
            Dictionary containing checkpoint data
        """
        filename = f"{experiment_name}_checkpoint_round_{round_num}.pt"
        filepath = os.path.join(self.directories['checkpoints'], filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
        checkpoint_data = torch.load(filepath, map_location='cpu')
        self.logger.info(f"Checkpoint loaded: {filepath}")
        
        return checkpoint_data
    
    def get_latest_checkpoint(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Latest checkpoint data or None if no checkpoints found
        """
        checkpoint_files = [f for f in os.listdir(self.directories['checkpoints'])
                           if f.startswith(f"{experiment_name}_checkpoint_")]
        
        if not checkpoint_files:
            return None
        
        # Sort by round number
        def extract_round_num(filename):
            try:
                return int(filename.split('_round_')[1].split('.')[0])
            except:
                return 0
        
        latest_file = max(checkpoint_files, key=extract_round_num)
        latest_round = extract_round_num(latest_file)
        
        return self.load_checkpoint(experiment_name, latest_round)
    
    def save_experimental_data(self, data: Dict[str, Any], 
                              experiment_name: str, data_type: str) -> str:
        """
        Save experimental data (metrics, results, etc.).
        
        Args:
            data: Data to save
            experiment_name: Name of the experiment
            data_type: Type of data (e.g., 'metrics', 'results', 'analysis')
            
        Returns:
            Full path to the saved data file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{experiment_name}_{data_type}_{timestamp}.json"
        filepath = os.path.join(self.directories['processed_data'], filename)
        
        # Add metadata
        data_with_metadata = {
            'experiment_name': experiment_name,
            'data_type': data_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data_with_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Experimental data saved: {filepath}")
        return filepath
    
    def load_experimental_data(self, filepath: str) -> Dict[str, Any]:
        """
        Load experimental data from file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            Loaded data dictionary
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Experimental data loaded: {filepath}")
        return data
    
    def save_numpy_array(self, array: np.ndarray, filename: str, 
                        data_category: str = 'raw_data') -> str:
        """
        Save numpy array to file.
        
        Args:
            array: Numpy array to save
            filename: Filename for the array
            data_category: Category of data ('raw_data' or 'processed_data')
            
        Returns:
            Full path to the saved file
        """
        filepath = os.path.join(self.directories[data_category], filename)
        np.save(filepath, array)
        self.logger.info(f"Numpy array saved: {filepath}")
        return filepath
    
    def load_numpy_array(self, filename: str, 
                        data_category: str = 'raw_data') -> np.ndarray:
        """
        Load numpy array from file.
        
        Args:
            filename: Filename of the array
            data_category: Category of data ('raw_data' or 'processed_data')
            
        Returns:
            Loaded numpy array
        """
        filepath = os.path.join(self.directories[data_category], filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Array file not found: {filepath}")
        
        array = np.load(filepath)
        self.logger.info(f"Numpy array loaded: {filepath}")
        return array
    
    def save_config(self, config: Dict[str, Any], config_name: str) -> str:
        """
        Save configuration dictionary to file.
        
        Args:
            config: Configuration dictionary
            config_name: Name for the configuration
            
        Returns:
            Full path to the saved config file
        """
        filename = f"{config_name}.json"
        filepath = os.path.join(self.directories['configs'], filename)
        
        config_with_metadata = {
            'config_name': config_name,
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_with_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved: {filepath}")
        return filepath
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Configuration dictionary
        """
        filename = f"{config_name}.json"
        filepath = os.path.join(self.directories['configs'], filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            config_data = json.load(f)
        
        self.logger.info(f"Configuration loaded: {filepath}")
        return config_data.get('config', config_data)
    
    def cleanup_old_files(self, days_old: int = 7, file_types: List[str] = None):
        """
        Clean up old files to save disk space.
        
        Args:
            days_old: Remove files older than this many days
            file_types: List of file types to clean ('checkpoints', 'logs', etc.)
        """
        if file_types is None:
            file_types = ['checkpoints', 'logs']
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        files_removed = 0
        
        for file_type in file_types:
            if file_type in self.directories:
                directory = self.directories[file_type]
                
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    
                    if os.path.isfile(filepath):
                        file_time = os.path.getmtime(filepath)
                        
                        if file_time < cutoff_time:
                            os.remove(filepath)
                            files_removed += 1
        
        self.logger.info(f"Cleanup completed: removed {files_removed} old files")
    
    def get_directory_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all managed directories.
        
        Returns:
            Dictionary containing directory information
        """
        info = {}
        
        for name, path in self.directories.items():
            if os.path.exists(path):
                files = os.listdir(path)
                total_size = sum(os.path.getsize(os.path.join(path, f)) 
                               for f in files if os.path.isfile(os.path.join(path, f)))
                
                info[name] = {
                    'path': path,
                    'num_files': len(files),
                    'total_size_mb': total_size / (1024 * 1024),
                    'exists': True
                }
            else:
                info[name] = {
                    'path': path,
                    'exists': False
                }
        
        return info


# Convenience functions for backward compatibility
def save_model(model: torch.nn.Module, filepath: str, 
               additional_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Convenience function to save a model.
    
    Args:
        model: PyTorch model to save
        filepath: Full path where to save the model
        additional_info: Additional information to save
        
    Returns:
        Path to saved file
    """
    save_data = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    if additional_info:
        save_data.update(additional_info)
    
    torch.save(save_data, filepath)
    return filepath


def load_model(model: torch.nn.Module, filepath: str) -> Dict[str, Any]:
    """
    Convenience function to load a model.
    
    Args:
        model: PyTorch model to load state into
        filepath: Full path to the model file
        
    Returns:
        Model metadata
    """
    save_data = torch.load(filepath, map_location='cpu')
    model.load_state_dict(save_data['model_state_dict'])
    
    return {k: v for k, v in save_data.items() if k != 'model_state_dict'}
