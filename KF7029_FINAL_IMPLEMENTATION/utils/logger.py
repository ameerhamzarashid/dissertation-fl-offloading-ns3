"""
Logger Setup - Centralized logging configuration
Provides structured logging for all system components
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                 level: int = logging.INFO, 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up a logger with consistent formatting and output options.
    
    Args:
        name: Logger name (usually module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_string: Optional custom format string
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Default format string
    if format_string is None:
        format_string = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_system_logging(log_directory: str = 'logs', log_level: int = logging.INFO):
    """
    Set up system-wide logging configuration.
    
    Args:
        log_directory: Directory for log files
        log_level: Global logging level
    """
    # Create log directory
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Generate timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Main system log
    main_log_file = os.path.join(log_directory, f'fl_system_{timestamp}.log')
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return main_log_file


def get_component_logger(component_name: str, log_directory: str = 'logs') -> logging.Logger:
    """
    Get a logger for a specific system component.
    
    Args:
        component_name: Name of the component
        log_directory: Directory for component-specific log files
        
    Returns:
        Component-specific logger
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_directory, f'{component_name}_{timestamp}.log')
    
    return setup_logger(component_name, log_file=log_file)


class PerformanceLogger:
    """
    Specialized logger for performance metrics and timing information.
    """
    
    def __init__(self, name: str, log_directory: str = 'logs'):
        """
        Initialize performance logger.
        
        Args:
            name: Logger name
            log_directory: Directory for log files
        """
        self.name = name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_directory, f'performance_{name}_{timestamp}.log')
        
        self.logger = setup_logger(
            f'Performance_{name}',
            log_file=log_file,
            format_string='[%(asctime)s] PERF - %(message)s'
        )
    
    def log_timing(self, operation: str, duration: float, additional_info: Optional[dict] = None):
        """
        Log timing information for an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            additional_info: Optional additional information
        """
        message = f"{operation}: {duration:.4f}s"
        
        if additional_info:
            info_str = ', '.join([f"{k}={v}" for k, v in additional_info.items()])
            message += f" | {info_str}"
        
        self.logger.info(message)
    
    def log_memory_usage(self, operation: str, memory_mb: float):
        """
        Log memory usage information.
        
        Args:
            operation: Name of the operation
            memory_mb: Memory usage in MB
        """
        self.logger.info(f"MEMORY {operation}: {memory_mb:.2f} MB")
    
    def log_communication_cost(self, round_num: int, bytes_transmitted: int):
        """
        Log communication cost information.
        
        Args:
            round_num: Federated learning round number
            bytes_transmitted: Number of bytes transmitted
        """
        mb_transmitted = bytes_transmitted / (1024 * 1024)
        self.logger.info(f"COMM Round {round_num}: {mb_transmitted:.2f} MB ({bytes_transmitted} bytes)")


class MetricsLogger:
    """
    Specialized logger for tracking ML metrics and experimental results.
    """
    
    def __init__(self, experiment_name: str, log_directory: str = 'logs'):
        """
        Initialize metrics logger.
        
        Args:
            experiment_name: Name of the experiment
            log_directory: Directory for log files
        """
        self.experiment_name = experiment_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_directory, f'metrics_{experiment_name}_{timestamp}.log')
        
        self.logger = setup_logger(
            f'Metrics_{experiment_name}',
            log_file=log_file,
            format_string='[%(asctime)s] METRICS - %(message)s'
        )
    
    def log_training_metrics(self, round_num: int, metrics: dict):
        """
        Log training metrics for a specific round.
        
        Args:
            round_num: Training round number
            metrics: Dictionary of metrics
        """
        metrics_str = ', '.join([f"{k}={v:.6f}" for k, v in metrics.items()])
        self.logger.info(f"Round {round_num}: {metrics_str}")
    
    def log_evaluation_results(self, evaluation_type: str, results: dict):
        """
        Log evaluation results.
        
        Args:
            evaluation_type: Type of evaluation (e.g., 'validation', 'test')
            results: Dictionary of evaluation results
        """
        results_str = ', '.join([f"{k}={v:.6f}" for k, v in results.items()])
        self.logger.info(f"EVAL {evaluation_type}: {results_str}")
    
    def log_experiment_config(self, config: dict):
        """
        Log experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("EXPERIMENT CONFIG:")
        for section, params in config.items():
            if isinstance(params, dict):
                params_str = ', '.join([f"{k}={v}" for k, v in params.items()])
                self.logger.info(f"  {section}: {params_str}")
            else:
                self.logger.info(f"  {section}: {params}")


def configure_experiment_logging(experiment_name: str, 
                                config: dict, 
                                log_directory: str = 'logs') -> tuple:
    """
    Configure logging for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
        log_directory: Directory for all log files
        
    Returns:
        Tuple of (main_logger, performance_logger, metrics_logger)
    """
    # Setup log directory
    experiment_log_dir = os.path.join(log_directory, experiment_name)
    if not os.path.exists(experiment_log_dir):
        os.makedirs(experiment_log_dir)
    
    # Create specialized loggers
    main_logger = setup_logger(f'Main_{experiment_name}', 
                              log_file=os.path.join(experiment_log_dir, 'main.log'))
    
    performance_logger = PerformanceLogger(experiment_name, experiment_log_dir)
    metrics_logger = MetricsLogger(experiment_name, experiment_log_dir)
    
    # Log experiment configuration
    metrics_logger.log_experiment_config(config)
    main_logger.info(f"Experiment '{experiment_name}' logging configured")
    
    return main_logger, performance_logger, metrics_logger
