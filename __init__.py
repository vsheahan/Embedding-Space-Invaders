"""
Embedding Space Invaders - Multi-Layer Detection System for LLM Prompt Injection

A multi-layer embedding analysis system that attempts to detect prompt injection
attacks by measuring geometric distances in transformer embedding space.

Status: Doesn't really work (96.9% and 100% FPR), but well-documented!

Key Features:
- Multi-layer transformer analysis (5 layers)
- Statistical anomaly detection (Mahalanobis, cosine similarity, PCA)
- Detection-first approach with optional mitigation
- Comprehensive logging and evaluation
- Spectacular failure metrics

Main Components:

1. EmbeddingSpaceInvaders: Multi-layer detection pipeline
2. ESInvadersHarness: Experiment runner and evaluator

Usage:
    # Basic detection
    from embedding_space_invaders import EmbeddingSpaceInvaders

    detector = EmbeddingSpaceInvaders(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        detection_mode="voting",
        min_anomalous_layers=2
    )
    detector.train_baseline(safe_prompts)
    scores = detector.score_prompt("User input here")

    # Run experiments
    from embedding_space_invaders import ESInvadersHarness

    harness = ESInvadersHarness(detector)
    results = harness.run_detection_experiment(test_prompts, labels)
"""

from .embedding_space_invaders import (
    EmbeddingSpaceInvaders,
    AggregatedScores,
    LayerScores,
    ThresholdConfig
)
from .es_invaders_harness import ESInvadersHarness
from .metrics import (
    compute_mahalanobis_distance,
    compute_cosine_similarity,
    compute_pca_projections,
    compute_baseline_statistics,
    compute_attack_centroid
)

__version__ = "1.0.0"
__all__ = [
    "EmbeddingSpaceInvaders",
    "ESInvadersHarness",
    "AggregatedScores",
    "LayerScores",
    "ThresholdConfig",
    "compute_mahalanobis_distance",
    "compute_cosine_similarity",
    "compute_pca_projections",
    "compute_baseline_statistics",
    "compute_attack_centroid"
]
