"""Attacks used in the experiments. Each is one plain function."""
import numpy as np


def flip_labels(y, fraction, seed=1337):
    """Flip the labels of a random `fraction` of a binary label array.

    Returns (y_poisoned, flipped_indices). `y` itself is not changed.
    """
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    y = np.asarray(y)
    n_flip = int(len(y) * fraction)
    rng = np.random.default_rng(seed)
    flipped = rng.choice(len(y), size=n_flip, replace=False) if n_flip else np.array([], dtype=int)
    y_poisoned = y.copy()
    y_poisoned[flipped] = 1 - y_poisoned[flipped]
    return y_poisoned, flipped
