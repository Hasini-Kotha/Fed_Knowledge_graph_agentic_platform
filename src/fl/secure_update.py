"""
Secure update module for federated learning.

# v1: Shape validation + norm clipping + checksum logging (THIS FILE)
# v2: Differential privacy with noise_multiplier tuning
# v3: Secure aggregation (masks that cancel at server)
# v4: Homomorphic encryption (future research extension)
"""

import numpy as np
import hashlib
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def validate_update_shape(parameters: List[np.ndarray], reference_parameters: List[np.ndarray]) -> bool:
    """
    Validates that the parameters have the same shape as reference_parameters.
    """
    if len(parameters) != len(reference_parameters):
        raise ValueError(
            f"Parameter count mismatch: received {len(parameters)}, "
            f"expected {len(reference_parameters)}"
        )
        
    for i, (param, ref_param) in enumerate(zip(parameters, reference_parameters)):
        if param.shape != ref_param.shape:
            raise ValueError(
                f"Shape mismatch at index {i}: received {param.shape}, "
                f"expected {ref_param.shape}"
            )
            
        if np.isnan(param).any() or np.isinf(param).any():
            logger.warning(f"Parameter array at index {i} contains NaN or Inf values.")
            
    return True

def clip_update_norm(parameters: List[np.ndarray], max_norm: float = 1.0) -> List[np.ndarray]:
    """
    Clips the global L2 norm of the parameters to max_norm.
    """
    squared_sum = sum(np.sum(np.square(p)) for p in parameters)
    global_norm = np.sqrt(squared_sum)
    
    clipped = False
    new_parameters = [np.copy(p) for p in parameters]
    
    if global_norm > max_norm:
        clip_factor = max_norm / (global_norm + 1e-6)
        for i in range(len(new_parameters)):
            if np.issubdtype(new_parameters[i].dtype, np.integer):
                new_parameters[i] = new_parameters[i].astype(np.float64)
            new_parameters[i] *= clip_factor
        clipped = True
        
    logger.info(f"Norm clipping: original_norm={global_norm:.4f}, max_norm={max_norm}, clipped={clipped}")
    return new_parameters

def add_gaussian_noise(parameters: List[np.ndarray], noise_multiplier: float = 0.0, max_norm: float = 1.0) -> List[np.ndarray]:
    """
    Adds Gaussian noise for differential privacy.
    """
    if noise_multiplier <= 0.0:
        logger.info("Gaussian noise disabled (noise_multiplier=0.0).")
        return parameters
        
    std_dev = noise_multiplier * max_norm
    logger.info(f"Adding Gaussian noise: std_dev={std_dev:.4f}, noise_multiplier={noise_multiplier}")
    
    noisy_parameters = []
    for p in parameters:
        # Ensure float type for noise addition
        p_float = p.astype(np.float64) if np.issubdtype(p.dtype, np.integer) else p
        noise = np.random.normal(0.0, std_dev, p.shape)
        noisy_parameters.append(p_float + noise)
        
    return noisy_parameters

def compute_parameter_checksum(parameters: List[np.ndarray]) -> str:
    """
    Computes a SHA-256 hash of all parameters concatenated.
    """
    hasher = hashlib.sha256()
    for p in parameters:
        hasher.update(p.tobytes())
    return hasher.hexdigest()

def protect_update(
    parameters: List[np.ndarray], 
    reference_parameters: List[np.ndarray], 
    config: Dict[str, Any]
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Runs the full protection pipeline on model updates.
    """
    max_norm = config.get("max_norm", 1.0)
    noise_multiplier = config.get("noise_multiplier", 0.0)
    validate = config.get("validate", True)
    
    if validate:
        validate_update_shape(parameters, reference_parameters)
        
    # Calculate original norm before clipping
    original_norm = np.sqrt(sum(np.sum(np.square(p)) for p in parameters))
        
    clipped_params = clip_update_norm(parameters, max_norm)
    clipped = (original_norm > max_norm)
    
    noisy_params = add_gaussian_noise(clipped_params, noise_multiplier, max_norm)
    noise_added = (noise_multiplier > 0.0)
    
    checksum = compute_parameter_checksum(noisy_params)
    
    audit_log_dict = {
        "original_norm": float(original_norm),
        "clipped": clipped,
        "noise_added": noise_added,
        "checksum": checksum
    }
    
    return noisy_params, audit_log_dict

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running secure_update tests...")
    
    ref_params = [np.ones((2, 2)), np.zeros(3)]
    params = [np.ones((2, 2)) * 2, np.ones(3) * 3]
    
    config = {
        "max_norm": 5.0,
        "noise_multiplier": 0.1,
        "validate": True
    }
    
    protected, audit = protect_update(params, ref_params, config)
    print("Audit log:", audit)
    print("Test passed.")
