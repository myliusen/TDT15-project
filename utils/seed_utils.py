# (Courtesy of Claude Sonnet 4.5)

def set_random_seed(seed=42):
    """Set random seed for reproducibility across all libraries (TensorFlow, Keras, PyTorch)"""
    import random
    import os
    import numpy as np
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seeds
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # For newer PyTorch versions
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

    # Set TensorFlow/Keras random seeds
    import tensorflow as tf
    import keras
    
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)
    
    # Enable deterministic operations
    tf.config.experimental.enable_op_determinism()
    
    # Set environment variables for full reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # For PyTorch CUDA determinism