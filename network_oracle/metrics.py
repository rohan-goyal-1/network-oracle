"""Small measurement helpers used by simulations and histories."""

from __future__ import annotations

import math


def binary_entropy(fraction: float) -> float:
    """Entropy in bits of a binary split, maximized at ``fraction == 0.5``."""
    if fraction <= 0.0 or fraction >= 1.0:
        return 0.0
    return -(
        fraction * math.log2(fraction)
        + (1.0 - fraction) * math.log2(1.0 - fraction)
    )
