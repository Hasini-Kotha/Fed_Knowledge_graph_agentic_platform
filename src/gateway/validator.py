"""validator.py — Weight submission validation before FedProx aggregation.

Checks (in order):
    1. Round number matches the server's current expected round.
    2. Model version string matches what the server expects.
    3. Number of parameter tensors matches the reference shape list.
    4. Each tensor has the correct shape.
    5. No NaN values in any tensor.
    6. No Inf values in any tensor.

Returns a (passed: bool, reason: str) tuple so the caller can log and respond.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference shapes for LiteFraudNet (input=64, hidden=64, emb=32)
# Must match Fed_model.py parameter order exactly.
#
# layer                       weight shape        bias shape
#  projection.0 (Linear)      (64, 64)            (64,)
#  projection.1 (LayerNorm)   (64,)               (64,)
#  res_block1.block.0         (64, 64)            (64,)
#  res_block1.block.1 (LN)    (64,)               (64,)
#  res_block1.block.4         (64, 64)            (64,)
#  res_block1.block.5 (LN)    (64,)               (64,)
#  res_block2  (same ×3)      same as block1
#  emb_head.0  (Linear)       (32, 64)            (32,)
#  emb_head.1  (LayerNorm)    (32,)               (32,)
#  clf_head.0  (Linear)       (16, 64)            (16,)
#  clf_head.3  (Linear)       (1, 16)             (1,)
# ---------------------------------------------------------------------------

EXPECTED_MODEL_VERSION = "LiteFraudNet-v1"

EXPECTED_SHAPES: List[tuple] = [
    # projection block
    (64, 64), (64,), (64,), (64,),
    # res_block1
    (64, 64), (64,), (64,), (64,), (64, 64), (64,), (64,), (64,),
    # res_block2 (identical structure)
    (64, 64), (64,), (64,), (64,), (64, 64), (64,), (64,), (64,),
    # embedding head
    (32, 64), (32,), (32,), (32,),
    # classification head
    (16, 64), (16,), (1, 16), (1,),
]


def validate_weights(
    weights: List[np.ndarray],
    round_num: int,
    model_version: str,
    current_round: int,
) -> Tuple[bool, str]:
    """Validate decrypted model weights before FedProx aggregation.

    Args:
        weights:       List of numpy float32 arrays (already decrypted).
        round_num:     Round number the client claims to be submitting for.
        model_version: Model version string from the client.
        current_round: Server-side current round counter.

    Returns:
        (True, "ok") on success, or (False, "<reason>") on failure.
    """

    # 1 — Round check
    if round_num != current_round:
        return False, (
            f"Round mismatch: server expects round {current_round}, "
            f"client submitted for round {round_num}."
        )

    # 2 — Model version check
    if model_version != EXPECTED_MODEL_VERSION:
        return False, (
            f"Model version mismatch: expected '{EXPECTED_MODEL_VERSION}', "
            f"got '{model_version}'."
        )

    # 3 — Parameter count check
    if len(weights) != len(EXPECTED_SHAPES):
        return False, (
            f"Parameter count mismatch: expected {len(EXPECTED_SHAPES)} tensors, "
            f"got {len(weights)}."
        )

    # 4, 5, 6 — Per-tensor shape, NaN, Inf checks
    for idx, (arr, expected_shape) in enumerate(zip(weights, EXPECTED_SHAPES)):
        # Handle both np.ndarray and nested list (for backwards compatibility)
        if not isinstance(arr, np.ndarray):
            try:
                arr = np.array(arr, dtype=np.float32)
            except Exception as exc:
                return False, f"Tensor {idx}: failed to convert to float32 array — {exc}."

        if arr.shape != expected_shape:
            return False, (
                f"Tensor {idx}: shape mismatch — "
                f"expected {expected_shape}, got {arr.shape}."
            )

        if np.any(np.isnan(arr)):
            return False, f"Tensor {idx}: contains NaN values."

        if np.any(np.isinf(arr)):
            return False, f"Tensor {idx}: contains Inf values."

    logger.info(
        "Weight validation passed | round=%d | version=%s | tensors=%d",
        round_num, model_version, len(weights),
    )
    return True, "ok"
