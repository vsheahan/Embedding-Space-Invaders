"""
Subspace Sentinel - Advanced Multi-Layer Detection System for LLM Prompt Injection

A comprehensive, detection-first pipeline that uses multi-layer analysis with
math-based metrics (Mahalanobis distance, cosine similarity, PCA residuals) to
identify adversarial prompts without destructive mitigation.

Key Features:
- Multi-layer transformer analysis (5+ layers)
- Statistical anomaly detection with auto-calibrated thresholds
- Detection-first philosophy (mitigation optional)
- Comprehensive logging and evaluation
- Modular design ready for expansion
- Low false-positive rate through multi-metric voting

Two main components:

1. SubspaceSentinel: Advanced multi-layer detection pipeline
2. SentinelHarness: Experiment runner and evaluator

Usage:
    # Basic detection
    from subspace_sentinel import SubspaceSentinel

    detector = SubspaceSentinel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        detection_mode="voting",
        min_anomalous_layers=2
    )
    detector.train_baseline(safe_prompts)
    scores = detector.score_prompt("User input here")

    # Run experiments
    from subspace_sentinel import SentinelHarness

    harness = SentinelHarness(detector)
    results = harness.run_detection_experiment(test_prompts, labels)
"""

from .subspace_sentinel import (
    SubspaceSentinel,
    AggregatedScores,
    LayerScores,
    ThresholdConfig
)
from .sentinel_harness import SentinelHarness
from .metrics import (
    compute_mahalanobis_distance,
    compute_cosine_similarity,
    compute_pca_projections,
    compute_baseline_statistics,
    compute_attack_centroid
)

__version__ = "1.0.0"
__all__ = [
    "SubspaceSentinel",
    "SentinelHarness",
    "AggregatedScores",
    "LayerScores",
    "ThresholdConfig",
    "compute_mahalanobis_distance",
    "compute_cosine_similarity",
    "compute_pca_projections",
    "compute_baseline_statistics",
    "compute_attack_centroid"
]
