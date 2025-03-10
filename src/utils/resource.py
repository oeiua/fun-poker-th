"""
Resource utilization management for poker AI training.
"""

import os
import logging
import platform
import multiprocessing
from typing import Dict, Any

import tensorflow as tf
import numpy as np


def setup_resources(config: Dict[str, Any]) -> None:
    """
    Set up resources for training.
    
    This function configures CPU, GPU, and memory usage
    based on the configuration.
    
    Args:
        config: Configuration dictionary
    """
    resource_config = config['resources']
    
    # CPU configuration
    cpu_threads = resource_config.get('cpu_threads', -1)
    if cpu_threads == -1:
        cpu_threads = multiprocessing.cpu_count()
    
    # Configure TensorFlow thread settings
    tf.config.threading.set_inter_op_parallelism_threads(cpu_threads)
    tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
    
    logging.info(f"CPU configuration: Using {cpu_threads} threads")
    
    # GPU configuration
    use_gpu = resource_config.get('use_gpu', True)
    
    if use_gpu:
        # Check if GPUs are available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Log GPU information
                logging.info(f"GPU configuration: {len(gpus)} GPUs available")
                for i, gpu in enumerate(gpus):
                    logging.info(f"  GPU {i}: {gpu.name}")
                
                # Set visible devices if specified
                visible_gpus = resource_config.get('visible_gpus', None)
                if visible_gpus is not None:
                    visible_devices = [gpus[i] for i in visible_gpus if i < len(gpus)]
                    tf.config.set_visible_devices(visible_devices, 'GPU')
                    logging.info(f"Using {len(visible_devices)} GPUs: {visible_gpus}")
                
                # Set memory limit if specified
                memory_limit = resource_config.get('memory_limit_gb', None)
                if memory_limit is not None:
                    memory_limit_bytes = memory_limit * 1024 * 1024 * 1024  # Convert GB to bytes
                    for gpu in gpus:
                        tf.config.set_logical_device_configuration(
                            gpu, 
                            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_bytes)]
                        )
                    logging.info(f"GPU memory limit set to {memory_limit} GB")
                
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                logging.error(f"GPU configuration error: {e}")
        else:
            logging.warning("No GPUs found. Using CPU for training.")
            # Fall back to CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        logging.info("GPU disabled in configuration. Using CPU for training.")
        # Disable GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Memory configuration
    memory_limit_gb = resource_config.get('memory_limit_gb', None)
    if memory_limit_gb:
        if platform.system() == 'Linux':
            # On Linux, use resource module to limit memory
            try:
                import resource
                memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024  # Convert GB to bytes
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
                logging.info(f"Process memory limit set to {memory_limit_gb} GB")
            except ImportError:
                logging.warning("Failed to import resource module for memory limiting")
        else:
            logging.warning(f"Memory limit of {memory_limit_gb} GB requested, but not supported on {platform.system()}")
    
    # Set TensorFlow to use mixed precision
    if use_gpu and resource_config.get('mixed_precision', True):
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logging.info("Using mixed precision (float16) for training")
    
    # Random seed for reproducibility
    random_seed = config.get('random_seed', None)
    if random_seed is not None:
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        logging.info(f"Random seed set to {random_seed}")
    
    # Log TensorFlow version
    logging.info(f"TensorFlow version: {tf.__version__}")
    logging.info(f"Numpy version: {np.__version__}")
    
    # Log system information
    logging.info(f"System: {platform.system()} {platform.release()}")
    logging.info(f"Python version: {platform.python_version()}")
    logging.info(f"CPU count: {multiprocessing.cpu_count()}")
    
    # Log available memory
    try:
        import psutil
        mem_info = psutil.virtual_memory()
        logging.info(f"Available memory: {mem_info.available / (1024**3):.2f} GB / {mem_info.total / (1024**3):.2f} GB")
    except ImportError:
        logging.warning("psutil not installed, cannot log memory information")
    
    logging.info("Resource setup complete")