import numpy as np
from typing import Optional, Tuple

def compute_ipw_weights(
    accept: np.ndarray,
    p_accept: np.ndarray,
    stabilize: bool = True,
    trim_quantiles: Optional[Tuple[float, float]] = (0.01, 0.99),
    cap_max: Optional[float] = None,
) -> np.ndarray:
    """
    Return IPW weights for ALL rows; non-accepted rows get 0.
    Compatible with Python < 3.10 (uses Optional[...] instead of | None).
    """
    accept = np.asarray(accept, dtype=int)
    p = np.clip(np.asarray(p_accept, dtype=float), 1e-6, 1 - 1e-6)

    w = np.zeros_like(p, dtype=float)
    mask_acc = (accept == 1)
    if not np.any(mask_acc):
        return w  # no accepted obs, nothing to weight

    numer = accept.mean() if stabilize else 1.0
    w_acc = numer / p[mask_acc]

    if trim_quantiles is not None:
        lo, hi = np.quantile(w_acc, trim_quantiles)
        w_acc = np.clip(w_acc, lo, hi)
    if cap_max is not None:
        w_acc = np.minimum(w_acc, cap_max)

    w[mask_acc] = w_acc
    return w

def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    denom = np.sum(w**2)
    return float((w.sum()**2) / (denom + 1e-12))
