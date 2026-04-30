"""
Secure update module for federated learning.

Protection pipeline:
  v1 (THIS FILE): Shape validation + delta-based norm clipping + optional
                  Gaussian noise (correct DP formulation) + checksum audit.
  v2: Tunable noise_multiplier with (epsilon, delta)-DP accounting.
  v3: Secure aggregation via cancelling masks.
  v4: Homomorphic encryption (future research).

KEY FIX: Clipping and noise are applied to the *weight delta*
  (local_weights - global_weights), NOT to the absolute weight vector.
  This is required for any meaningful DP guarantee.
"""

import hashlib
import logging
import numpy as np
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


def validate_update_shape(
    parameters: List[np.ndarray],
    reference_parameters: List[np.ndarray],
) -> bool:
    """Validate shape and numerical health of parameters vs reference.

    Raises ValueError on shape mismatch.
    Logs a warning (does NOT reject) on NaN/Inf — caller decides policy.
    """
    if len(parameters) != len(reference_parameters):
        raise ValueError(
            f"Parameter count mismatch: received {len(parameters)}, "
            f"expected {len(reference_parameters)}."
        )
    for i, (p, ref) in enumerate(zip(parameters, reference_parameters)):
        if p.shape != ref.shape:
            raise ValueError(
                f"Shape mismatch at index {i}: received {p.shape}, "
                f"expected {ref.shape}."
            )
        if np.isnan(p).any() or np.isinf(p).any():
            logger.warning(
                "Parameter array at index %d contains NaN/Inf values.", i
            )
    return True


def clip_update_norm(
    delta: List[np.ndarray],
    max_norm: float = 1.0,
) -> List[np.ndarray]:
    """Clip the L2 norm of a *delta* (difference) vector to max_norm.

    Args:
        delta: List of numpy arrays representing weight differences.
        max_norm: Maximum allowed global L2 norm.

    Returns:
        Clipped delta with the same shapes, always float64.
    """
    squared_sum = sum(np.sum(np.square(d.astype(np.float64))) for d in delta)
    global_norm = float(np.sqrt(squared_sum))
    clipped = global_norm > max_norm

    new_delta = [d.astype(np.float64) for d in delta]
    if clipped:
        clip_factor = max_norm / (global_norm + 1e-9)
        new_delta = [d * clip_factor for d in new_delta]

    logger.info(
        "Norm clipping: delta_norm=%.4f, max_norm=%.4f, clipped=%s",
        global_norm, max_norm, clipped,
    )
    return new_delta


def add_gaussian_noise(
    delta: List[np.ndarray],
    noise_multiplier: float = 0.0,
    max_norm: float = 1.0,
) -> List[np.ndarray]:
    """Add calibrated Gaussian noise for differential privacy.

    std_dev = noise_multiplier × max_norm (standard DP-SGD formulation).
    No-op when noise_multiplier <= 0.

    Args:
        delta: Clipped weight-delta arrays.
        noise_multiplier: DP noise scale (σ / C).
        max_norm: Clipping threshold C.

    Returns:
        Noisy delta arrays.
    """
    if noise_multiplier <= 0.0:
        logger.info("Gaussian noise disabled (noise_multiplier=%.4f).", noise_multiplier)
        return delta

    std_dev = noise_multiplier * max_norm
    logger.info(
        "Adding Gaussian noise: std_dev=%.4f (noise_multiplier=%.4f, max_norm=%.4f).",
        std_dev, noise_multiplier, max_norm,
    )
    return [d + np.random.normal(0.0, std_dev, d.shape) for d in delta]


def compute_parameter_checksum(parameters: List[np.ndarray]) -> str:
    """Compute SHA-256 of all parameter arrays for audit logging."""
    hasher = hashlib.sha256()
    for p in parameters:
        hasher.update(p.tobytes())
    return hasher.hexdigest()


def protect_update(
    updated_parameters: List[np.ndarray],
    reference_parameters: List[np.ndarray],
    config: Dict[str, Any],
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Run the full protection pipeline on local model updates.

    Correct DP-FL formulation:
        1. Compute delta = updated_weights - global_weights
        2. Clip L2 norm of delta to max_norm
        3. Optionally add Gaussian noise to clipped delta
        4. Reconstruct: protected_weights = global_weights + noisy_delta

    Args:
        updated_parameters: Weights after local training.
        reference_parameters: Global weights received from server (round start).
        config: Dict with keys max_norm, noise_multiplier, validate (bool).

    Returns:
        (protected_parameters, audit_log_dict)
    """
    max_norm: float = float(config.get("max_norm", 1.0))
    noise_multiplier: float = float(config.get("noise_multiplier", 0.0))
    do_validate: bool = bool(config.get("validate", True))

    if do_validate:
        validate_update_shape(updated_parameters, reference_parameters)

    # Step 1: Compute weight delta
    delta = [
        u.astype(np.float64) - r.astype(np.float64)
        for u, r in zip(updated_parameters, reference_parameters)
    ]
    original_delta_norm = float(np.sqrt(sum(np.sum(np.square(d)) for d in delta)))

    # Step 2: Clip delta
    clipped_delta = clip_update_norm(delta, max_norm)
    clipped = original_delta_norm > max_norm

    # Step 3: Add noise to clipped delta
    noisy_delta = add_gaussian_noise(clipped_delta, noise_multiplier, max_norm)
    noise_added = noise_multiplier > 0.0

    # Step 4: Reconstruct protected parameters
    protected_params = [
        (r.astype(np.float64) + nd).astype(u.dtype)
        for u, r, nd in zip(updated_parameters, reference_parameters, noisy_delta)
    ]

    checksum = compute_parameter_checksum(protected_params)
    audit_log = {
        "original_delta_norm": original_delta_norm,
        "clipped": clipped,
        "noise_added": noise_added,
        "checksum": checksum,
    }
    logger.info("protect_update audit: %s", audit_log)
    return protected_params, audit_log


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ref = [np.ones((2, 2), dtype=np.float32), np.zeros(3, dtype=np.float32)]
    updated = [np.ones((2, 2), dtype=np.float32) * 2, np.ones(3, dtype=np.float32)]
    cfg = {"max_norm": 1.0, "noise_multiplier": 0.1, "validate": True}
    protected, audit = protect_update(updated, ref, cfg)
    print("Audit:", audit)
    print("Self-test passed.")
