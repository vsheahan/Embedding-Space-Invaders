"""
Metrics and Statistical Utilities for Layered Defense

Provides functions for computing various anomaly detection metrics:
- Mahalanobis distance
- Cosine similarity
- Perplexity
- PCA projection magnitudes
"""

import torch
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_mahalanobis_distance(
    embedding: np.ndarray,
    mean: np.ndarray,
    cov_inv: np.ndarray
) -> float:
    """
    Compute Mahalanobis distance of embedding from distribution.

    Args:
        embedding: Embedding vector [embed_dim]
        mean: Mean vector of baseline distribution [embed_dim]
        cov_inv: Inverse covariance matrix [embed_dim, embed_dim]

    Returns:
        Mahalanobis distance (float)
    """
    diff = embedding - mean
    distance = np.sqrt(diff @ cov_inv @ diff)
    return float(distance)


def compute_cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector

    Returns:
        Cosine similarity in [-1, 1]
    """
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    return float(similarity)


def compute_perplexity(
    model,
    tokenizer,
    text: str,
    device: str = "cpu",
    max_length: int = 512
) -> float:
    """
    Compute perplexity of text under the model.

    Lower perplexity = more "expected" by the model.
    Higher perplexity = more surprising/anomalous.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        text: Text to compute perplexity for
        device: Device to use
        max_length: Maximum sequence length

    Returns:
        Perplexity value
    """
    if not text or len(text.strip()) == 0:
        return float('inf')

    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    )

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            # Cross-entropy loss
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

    except Exception as e:
        logger.warning(f"Error computing perplexity: {e}")
        return float('inf')


def compute_pca_projections(
    embedding: np.ndarray,
    pca_components: np.ndarray
) -> np.ndarray:
    """
    Compute projection magnitudes onto PCA components.

    Args:
        embedding: Input embedding [embed_dim]
        pca_components: PCA components [n_components, embed_dim]

    Returns:
        Projection magnitudes [n_components]
    """
    # Project onto each component
    projections = embedding @ pca_components.T
    return projections


def compute_baseline_statistics(
    embeddings: np.ndarray,
    regularization: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, covariance, and inverse covariance from embeddings.

    Args:
        embeddings: Array of embeddings [n_samples, embed_dim]
        regularization: Regularization factor for covariance inversion

    Returns:
        Tuple of (mean, covariance, inverse_covariance)
    """
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    # Compute covariance with regularization
    cov = np.cov(centered, rowvar=False)

    # Add regularization to diagonal for numerical stability
    cov_reg = cov + regularization * np.eye(cov.shape[0])

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Covariance matrix singular, using pseudo-inverse")
        cov_inv = np.linalg.pinv(cov_reg)

    return mean, cov, cov_inv


def compute_attack_centroid(
    attack_embeddings: np.ndarray,
    baseline_mean: np.ndarray
) -> np.ndarray:
    """
    Compute centroid direction from baseline to attacks.

    This represents the "average attack direction" in embedding space.

    Args:
        attack_embeddings: Array of attack embeddings [n_attacks, embed_dim]
        baseline_mean: Mean of baseline (safe) embeddings [embed_dim]

    Returns:
        Normalized centroid direction vector [embed_dim]
    """
    attack_mean = np.mean(attack_embeddings, axis=0)
    centroid_direction = attack_mean - baseline_mean

    # Normalize
    norm = np.linalg.norm(centroid_direction)
    if norm > 0:
        centroid_direction = centroid_direction / norm

    return centroid_direction
