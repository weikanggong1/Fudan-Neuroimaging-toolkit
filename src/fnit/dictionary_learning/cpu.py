"""Array-based dictionary learning using scikit-learn's CPU reference."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning


def fit_dicl(projected: Mapping[str, np.ndarray], dicl_dim: int,
             max_iter: int = 1000, random_state: int = 0) -> dict[str, np.ndarray]:
    """Run the notebook's sklearn MiniBatchDictionaryLearning on each modality."""
    if not projected or dicl_dim < 2 or max_iter < 1:
        raise ValueError("DicL requires modalities, dicl_dim >= 2 and max_iter >= 1")
    result = {}
    for name, data in projected.items():
        if data.shape[0] < dicl_dim:
            raise ValueError(f"Mask has fewer voxels than dicl_dim: {name}")
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 0.1
        samples = (data - mean) / std
        learner = MiniBatchDictionaryLearning(
            n_components=dicl_dim, max_iter=max_iter, batch_size=32,
            transform_n_nonzero_coefs=max(1, int(dicl_dim * 0.15)),
            random_state=random_state,
        )
        dictionary = learner.fit(samples).components_.T
        dictionary -= dictionary.mean(axis=0, keepdims=True)
        scale = np.sqrt(np.mean(dictionary ** 2))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError(f"Degenerate dictionary: {name}")
        result[name] = (dictionary / scale).T
    return result
