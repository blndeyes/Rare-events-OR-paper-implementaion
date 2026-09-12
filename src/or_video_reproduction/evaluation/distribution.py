"""Distribution-level metric calculations from already extracted model outputs."""

from __future__ import annotations

import numpy as np


def _matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def frechet_distance(reference_embeddings: np.ndarray, generated_embeddings: np.ndarray) -> float:
    """Frechet distance between two embedding sets using a stable PSD formulation."""

    reference = np.asarray(reference_embeddings, dtype=np.float64)
    generated = np.asarray(generated_embeddings, dtype=np.float64)
    if reference.ndim != 2 or generated.ndim != 2 or reference.shape[1] != generated.shape[1]:
        raise ValueError("Embedding inputs must be [N,D] arrays with the same D")
    if len(reference) < 2 or len(generated) < 2:
        raise ValueError("Frechet distance requires at least two samples in each set")
    mean_difference = reference.mean(axis=0) - generated.mean(axis=0)
    covariance_reference = np.cov(reference, rowvar=False)
    covariance_generated = np.cov(generated, rowvar=False)
    sqrt_reference = _matrix_sqrt_psd(covariance_reference)
    middle = sqrt_reference @ covariance_generated @ sqrt_reference
    covariance_mean = _matrix_sqrt_psd(middle)
    value = mean_difference @ mean_difference + np.trace(
        covariance_reference + covariance_generated - 2.0 * covariance_mean
    )
    return float(max(0.0, value))


def inception_score(
    values: np.ndarray, *, splits: int = 10, from_logits: bool = True
) -> dict[str, float | int]:
    """Compute Inception Score from class logits or probabilities."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or values.shape[1] < 2:
        raise ValueError("Inception input must have shape [N,C] with C >= 2")
    if not 1 <= splits <= len(values):
        raise ValueError("splits must be between 1 and the number of samples")
    if from_logits:
        shifted = values - values.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
    else:
        if np.any(values < 0) or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Probability rows must be non-negative and sum to one")
        probabilities = values
    scores: list[float] = []
    epsilon = np.finfo(np.float64).tiny
    for part in np.array_split(probabilities, splits):
        marginal = part.mean(axis=0, keepdims=True)
        divergence = np.sum(part * (np.log(part + epsilon) - np.log(marginal + epsilon)), axis=1)
        scores.append(float(np.exp(np.mean(divergence))))
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "splits": splits}
