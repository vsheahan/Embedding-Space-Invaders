"""
Embedding Space Invaders for LLM Prompt Injection Detection

A comprehensive, detection-first pipeline that uses multi-layer analysis with
math-based metrics (Mahalanobis distance, cosine similarity, PCA residuals) to
identify adversarial prompts without destructive mitigation.

Key Features:
1. Multi-layer metric aggregation from transformer layers
2. Automatic threshold calibration from baseline prompts
3. Comprehensive detection logging with anomaly explanations
4. Optional selective mitigation for extreme outliers only
5. Modular design ready for expansion
6. Low false-positive rate through statistical validation

Usage:
    detector = EmbeddingSpaceInvaders(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    detector.train_baseline(safe_prompts)
    scores = detector.score_prompt(test_prompt)
    results = detector.evaluate_dataset(test_prompts, labels)
"""

import json
import logging
import os
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from .metrics import (
        compute_mahalanobis_distance,
        compute_cosine_similarity,
        compute_pca_projections,
        compute_baseline_statistics,
        compute_attack_centroid
    )
except ImportError:
    from metrics import (
        compute_mahalanobis_distance,
        compute_cosine_similarity,
        compute_pca_projections,
        compute_baseline_statistics,
        compute_attack_centroid
    )

logger = logging.getLogger(__name__)


@dataclass
class LayerScores:
    """Scores from a single transformer layer."""
    layer_idx: int
    mahalanobis_distance: float
    cosine_similarity: float
    pca_residual: float  # Residual after projecting onto top PCA components
    top_pca_projection: float  # Maximum projection magnitude


@dataclass
class AggregatedScores:
    """Aggregated detection scores across all layers."""
    # Aggregated metrics
    max_mahalanobis: float
    min_cosine_similarity: float
    max_pca_residual: float
    mean_mahalanobis: float
    mean_cosine_similarity: float

    # Detection flags
    is_anomalous: bool
    confidence_score: float  # 0-1, higher = more confident it's an attack
    anomaly_reasons: List[str]
    exceeds_extreme_threshold: bool

    # Layer-wise breakdown
    layer_scores: List[LayerScores]
    num_anomalous_layers: int


@dataclass
class ThresholdConfig:
    """Configuration for detection thresholds."""
    # Per-metric thresholds
    mahalanobis_threshold: float
    cosine_similarity_threshold: float
    pca_residual_threshold: float

    # Extreme outlier threshold (for mitigation)
    extreme_mahalanobis_threshold: float

    # Aggregation parameters
    anomaly_layer_count_threshold: int  # Min layers that must be anomalous


class EmbeddingSpaceInvaders:
    """
    Full-scale detection pipeline with multi-layer analysis.

    This system extracts embeddings from multiple transformer layers and computes
    anomaly detection metrics at each layer. Scores are aggregated to produce a
    final detection decision with high confidence and low false positives.

    Detection Strategy:
    - Extract hidden states from multiple transformer layers
    - Compute Mahalanobis distance, cosine similarity, and PCA residuals per layer
    - Aggregate metrics using statistical combination (max, mean, voting)
    - Flag anomalies only when multiple layers agree (reduces false positives)
    - Optional mitigation for extreme outliers only

    Recommended Usage:
    1. Train on 100+ diverse safe prompts for robust baseline
    2. Use detection-only mode by default (apply_mitigation=False)
    3. Enable mitigation only for high-stakes applications
    4. Tune thresholds based on validation set false positive rate
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None,
        # Layer selection
        layer_indices: Optional[List[int]] = None,  # None = auto-select layers
        # Threshold configuration (None = auto-compute)
        mahalanobis_std_multiplier: float = 3.0,
        cosine_similarity_threshold: float = 0.6,
        pca_residual_std_multiplier: float = 3.0,
        extreme_outlier_multiplier: float = 3.0,
        # Detection parameters
        min_anomalous_layers: int = 2,  # Require multiple layers to agree
        detection_mode: str = "voting",  # "voting", "max", "mean"
        # Mitigation parameters (CONSERVATIVE)
        mitigation_alpha: float = 0.05,
        mitigation_layers: Optional[List[int]] = None,  # Which layers to mitigate
        # PCA parameters
        n_pca_components: int = 10,
        pca_reconstruction_components: int = 5,  # For residual computation
        # Logging
        log_to_file: bool = False,
        log_file_path: Optional[str] = None
    ):
        """
        Initialize EmbeddingSpaceInvaders system.

        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detect)

            Layer Selection:
                layer_indices: List of layer indices to analyze
                    - None: Auto-select layers (early, middle, late)
                    - Example: [0, 5, 10, 15, 20] for specific layers

            Threshold Configuration (None = auto-compute from baseline):
                mahalanobis_std_multiplier: Multiplier for Mahalanobis threshold
                    - threshold = mean + multiplier * std
                    - Recommended: 2.0-4.0 (higher = fewer false positives)
                cosine_similarity_threshold: Minimum similarity to baseline
                    - Recommended: 0.5-0.7
                pca_residual_std_multiplier: Multiplier for PCA residual threshold
                    - Recommended: 2.0-4.0
                extreme_outlier_multiplier: Multiplier for mitigation threshold
                    - Only apply mitigation if Mahalanobis > threshold * multiplier
                    - Recommended: 2.0-3.0

            Detection Parameters:
                min_anomalous_layers: Minimum layers that must flag anomaly
                    - Higher = fewer false positives, may miss subtle attacks
                    - Recommended: 2-3 for balanced detection
                detection_mode: How to aggregate layer scores
                    - "voting": Count layers exceeding thresholds
                    - "max": Use maximum metric across layers
                    - "mean": Use average metric across layers

            Mitigation Parameters (CONSERVATIVE):
                mitigation_alpha: Dampening factor (0=none, 1=full)
                    - Recommended: 0.05-0.1 (very gentle)
                mitigation_layers: Which layers to apply mitigation to
                    - None: Apply to all analyzed layers

            PCA Parameters:
                n_pca_components: Number of PCA components to fit
                pca_reconstruction_components: Components for residual computation
                    - Residual = ||x - PCA_k(x)|| measures reconstruction error
        """
        logger.info(f"Initializing EmbeddingSpaceInvaders with model: {model_name}")

        # Device setup
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine layer indices
        num_layers = self.model.config.num_hidden_layers
        if layer_indices is None:
            # Auto-select: first, quarter, half, three-quarters, last
            self.layer_indices = [
                0,
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4,
                num_layers - 1
            ]
        else:
            self.layer_indices = layer_indices

        logger.info(f"Analyzing layers: {self.layer_indices} (total: {num_layers})")

        # Threshold configuration
        self.mahalanobis_std_multiplier = mahalanobis_std_multiplier
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.pca_residual_std_multiplier = pca_residual_std_multiplier
        self.extreme_outlier_multiplier = extreme_outlier_multiplier

        # Detection parameters
        self.min_anomalous_layers = min_anomalous_layers
        self.detection_mode = detection_mode

        # Mitigation parameters
        self.mitigation_alpha = mitigation_alpha
        self.mitigation_layers = mitigation_layers or self.layer_indices

        # PCA parameters
        self.n_pca_components = n_pca_components
        self.pca_reconstruction_components = pca_reconstruction_components

        # Baseline statistics (per layer)
        self.layer_baselines: Dict[int, Dict[str, Any]] = {}
        self.layer_pcas: Dict[int, PCA] = {}
        self.thresholds: Optional[ThresholdConfig] = None
        self.attack_centroids: Dict[int, np.ndarray] = {}  # For mitigation

        # Logging
        self.log_to_file = log_to_file
        self.log_file_path = log_file_path or "embedding_space_invaders.jsonl"

        logger.info(f"Device: {self.device}")
        logger.info(f"Detection mode: {detection_mode}")
        logger.info(f"Min anomalous layers: {min_anomalous_layers}")
        logger.info(f"Mitigation alpha: {mitigation_alpha}")

    def extract_multilayer_embeddings(
        self,
        prompt: str,
        max_length: int = 512
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from multiple transformer layers.

        Args:
            prompt: Input text
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping layer_idx -> pooled embedding
        """
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)

        embeddings = {}
        mask_expanded = attention_mask.unsqueeze(-1).float()

        for layer_idx in self.layer_indices:
            layer_hidden = hidden_states[layer_idx]
            # Mean pooling
            masked_sum = (layer_hidden * mask_expanded).sum(dim=1)
            mask_sum = mask_expanded.sum(dim=1)
            pooled = (masked_sum / mask_sum).cpu().numpy()[0]
            embeddings[layer_idx] = pooled

        return embeddings

    def extract_batch_multilayer_embeddings(
        self,
        prompts: List[str],
        batch_size: int = 8
    ) -> Dict[int, np.ndarray]:
        """
        Extract embeddings from multiple prompts efficiently.

        Args:
            prompts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping layer_idx -> array of embeddings [n_prompts, hidden_dim]
        """
        layer_embeddings = defaultdict(list)

        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting embeddings"):
            batch = prompts[i:i+batch_size]
            for prompt in batch:
                embs = self.extract_multilayer_embeddings(prompt)
                for layer_idx, emb in embs.items():
                    layer_embeddings[layer_idx].append(emb)

        # Convert to arrays
        return {
            layer_idx: np.array(embs)
            for layer_idx, embs in layer_embeddings.items()
        }

    def train_baseline(
        self,
        safe_prompts: List[str],
        regularization: float = 1e-6,
        auto_tune_thresholds: bool = True
    ):
        """
        Train baseline model from safe prompts with automatic threshold tuning.

        This method:
        1. Extracts embeddings from all selected layers
        2. Computes mean, covariance, and inverse covariance per layer
        3. Fits PCA models per layer
        4. Automatically computes detection thresholds from baseline statistics

        Args:
            safe_prompts: List of known-safe prompts (recommend 100+)
            regularization: Regularization for covariance inversion
            auto_tune_thresholds: Automatically compute thresholds from data
        """
        n_samples = len(safe_prompts)

        if n_samples < 50:
            logger.warning(
                f"⚠️  Only {n_samples} safe prompts provided. "
                f"Recommend 100+ for robust baseline. "
                f"Thresholds may be unreliable."
            )
        elif n_samples >= 100:
            logger.info(f"✓ Good baseline size: {n_samples} prompts")

        logger.info(f"Training baseline from {n_samples} safe prompts...")

        # Extract embeddings from all layers
        layer_embeddings = self.extract_batch_multilayer_embeddings(safe_prompts)

        # Train per-layer baselines
        for layer_idx, embeddings in layer_embeddings.items():
            logger.info(f"Training baseline for layer {layer_idx}...")

            # Compute baseline statistics
            mean, cov, cov_inv = compute_baseline_statistics(
                embeddings, regularization=regularization
            )

            # Fit PCA
            n_components = min(
                self.n_pca_components,
                n_samples,
                embeddings.shape[1]
            )
            pca = PCA(n_components=n_components)
            pca.fit(embeddings)

            variance_explained = pca.explained_variance_ratio_.sum()
            logger.info(
                f"  Layer {layer_idx}: PCA with {n_components} components, "
                f"variance explained: {variance_explained:.2%}"
            )

            # Store
            self.layer_baselines[layer_idx] = {
                "mean": mean,
                "cov": cov,
                "cov_inv": cov_inv,
                "embeddings": embeddings  # Keep for threshold computation
            }
            self.layer_pcas[layer_idx] = pca

        # Auto-tune thresholds
        if auto_tune_thresholds:
            self._compute_auto_thresholds()

        logger.info("Baseline training complete")

    def _compute_auto_thresholds(self):
        """
        Automatically compute detection thresholds from baseline distribution.

        For each layer:
        - Compute Mahalanobis distances for baseline prompts
        - Compute cosine similarities
        - Compute PCA residuals

        Then aggregate across layers to set global thresholds.
        """
        logger.info("Computing auto-tuned thresholds from baseline...")

        all_mahal_dists = []
        all_cosine_sims = []
        all_pca_residuals = []

        for layer_idx, baseline in self.layer_baselines.items():
            embeddings = baseline["embeddings"]
            mean = baseline["mean"]
            cov_inv = baseline["cov_inv"]
            pca = self.layer_pcas[layer_idx]

            # Compute metrics for baseline prompts
            for emb in embeddings:
                # Mahalanobis distance
                mahal = compute_mahalanobis_distance(emb, mean, cov_inv)
                all_mahal_dists.append(mahal)

                # Cosine similarity
                cos_sim = compute_cosine_similarity(emb, mean)
                all_cosine_sims.append(cos_sim)

                # PCA residual
                pca_proj = pca.transform([emb])[0][:self.pca_reconstruction_components]
                pca_recon = pca.inverse_transform(
                    np.pad(pca_proj, (0, pca.n_components_ - len(pca_proj)))
                )
                residual = np.linalg.norm(emb - pca_recon)
                all_pca_residuals.append(residual)

        # Compute statistics
        mahal_mean = np.mean(all_mahal_dists)
        mahal_std = np.std(all_mahal_dists)
        pca_residual_mean = np.mean(all_pca_residuals)
        pca_residual_std = np.std(all_pca_residuals)

        # Set thresholds
        mahalanobis_threshold = mahal_mean + self.mahalanobis_std_multiplier * mahal_std
        pca_residual_threshold = (
            pca_residual_mean + self.pca_residual_std_multiplier * pca_residual_std
        )
        extreme_mahalanobis_threshold = (
            mahalanobis_threshold * self.extreme_outlier_multiplier
        )

        self.thresholds = ThresholdConfig(
            mahalanobis_threshold=mahalanobis_threshold,
            cosine_similarity_threshold=self.cosine_similarity_threshold,
            pca_residual_threshold=pca_residual_threshold,
            extreme_mahalanobis_threshold=extreme_mahalanobis_threshold,
            anomaly_layer_count_threshold=self.min_anomalous_layers
        )

        logger.info(f"Auto-tuned thresholds:")
        logger.info(f"  Mahalanobis: {mahalanobis_threshold:.2f}")
        logger.info(f"  Cosine similarity: {self.cosine_similarity_threshold:.3f}")
        logger.info(f"  PCA residual: {pca_residual_threshold:.2f}")
        logger.info(f"  Extreme outlier (mitigation): {extreme_mahalanobis_threshold:.2f}")
        logger.info(f"  Min anomalous layers: {self.min_anomalous_layers}")

    def learn_attack_patterns(
        self,
        attack_prompts: List[str]
    ):
        """
        Learn attack patterns for selective mitigation (optional).

        Only needed if you plan to use mitigation mode.

        Args:
            attack_prompts: List of known attack prompts
        """
        logger.info(f"Learning attack patterns from {len(attack_prompts)} attacks...")

        if not self.layer_baselines:
            raise ValueError("Must call train_baseline() before learn_attack_patterns()")

        # Extract attack embeddings
        layer_embeddings = self.extract_batch_multilayer_embeddings(attack_prompts)

        # Compute attack centroids per layer
        for layer_idx, embeddings in layer_embeddings.items():
            baseline_mean = self.layer_baselines[layer_idx]["mean"]
            centroid = compute_attack_centroid(embeddings, baseline_mean)
            self.attack_centroids[layer_idx] = centroid

        logger.info("Attack patterns learned")
        logger.info(f"Learned centroids for {len(self.attack_centroids)} layers")

    def tune_thresholds_with_attacks(
        self,
        safe_prompts: List[str],
        attack_prompts: List[str],
        target_fpr: float = 0.05,
        target_tpr: float = 0.95
    ):
        """
        Tune detection thresholds using both safe and attack training data.

        This finds optimal thresholds that balance false positive and true positive rates
        by computing metrics on training data and searching for best operating point.

        Args:
            safe_prompts: Safe training prompts
            attack_prompts: Attack training prompts
            target_fpr: Target false positive rate (default: 5%)
            target_tpr: Target true positive rate (default: 95%)
        """
        logger.info(f"Tuning thresholds with {len(safe_prompts)} safe and {len(attack_prompts)} attack prompts...")

        if not self.layer_baselines:
            raise ValueError("Must call train_baseline() before tune_thresholds_with_attacks()")

        # Extract embeddings for both safe and attack prompts
        safe_layer_embs = self.extract_batch_multilayer_embeddings(safe_prompts)
        attack_layer_embs = self.extract_batch_multilayer_embeddings(attack_prompts)

        # Compute metrics for all prompts
        safe_metrics = []  # List of (mahal, cos_sim, pca_res) tuples
        attack_metrics = []

        for layer_idx in self.layer_indices:
            baseline = self.layer_baselines[layer_idx]
            mean = baseline["mean"]
            cov_inv = baseline["cov_inv"]
            pca = self.layer_pcas[layer_idx]

            # Safe prompts
            for emb in safe_layer_embs[layer_idx]:
                mahal = compute_mahalanobis_distance(emb, mean, cov_inv)
                cos_sim = compute_cosine_similarity(emb, mean)

                pca_proj = pca.transform([emb])[0][:self.pca_reconstruction_components]
                pca_recon = pca.inverse_transform(
                    np.pad(pca_proj, (0, pca.n_components_ - len(pca_proj)))
                )
                residual = np.linalg.norm(emb - pca_recon)

                safe_metrics.append((mahal, cos_sim, residual))

            # Attack prompts
            for emb in attack_layer_embs[layer_idx]:
                mahal = compute_mahalanobis_distance(emb, mean, cov_inv)
                cos_sim = compute_cosine_similarity(emb, mean)

                pca_proj = pca.transform([emb])[0][:self.pca_reconstruction_components]
                pca_recon = pca.inverse_transform(
                    np.pad(pca_proj, (0, pca.n_components_ - len(pca_proj)))
                )
                residual = np.linalg.norm(emb - pca_recon)

                attack_metrics.append((mahal, cos_sim, residual))

        # Convert to arrays for analysis
        safe_metrics = np.array(safe_metrics)
        attack_metrics = np.array(attack_metrics)

        # Find optimal thresholds for each metric
        # Mahalanobis: higher is more anomalous
        safe_mahal = safe_metrics[:, 0]
        attack_mahal = attack_metrics[:, 0]

        # Try different percentiles to find best threshold
        best_mahal_thresh = self._find_optimal_threshold(
            safe_mahal, attack_mahal, higher_is_attack=True, target_fpr=target_fpr, target_tpr=target_tpr
        )

        # Cosine similarity: lower is more anomalous
        safe_cos = safe_metrics[:, 1]
        attack_cos = attack_metrics[:, 1]

        best_cos_thresh = self._find_optimal_threshold(
            safe_cos, attack_cos, higher_is_attack=False, target_fpr=target_fpr, target_tpr=target_tpr
        )

        # PCA residual: higher is more anomalous
        safe_pca = safe_metrics[:, 2]
        attack_pca = attack_metrics[:, 2]

        best_pca_thresh = self._find_optimal_threshold(
            safe_pca, attack_pca, higher_is_attack=True, target_fpr=target_fpr, target_tpr=target_tpr
        )

        # Update thresholds
        old_thresholds = self.thresholds
        self.thresholds = ThresholdConfig(
            mahalanobis_threshold=best_mahal_thresh,
            cosine_similarity_threshold=best_cos_thresh,
            pca_residual_threshold=best_pca_thresh,
            extreme_mahalanobis_threshold=best_mahal_thresh * self.extreme_outlier_multiplier,
            anomaly_layer_count_threshold=self.min_anomalous_layers
        )

        logger.info("Threshold tuning complete!")
        logger.info(f"  Mahalanobis: {old_thresholds.mahalanobis_threshold:.2f} → {best_mahal_thresh:.2f}")
        logger.info(f"  Cosine similarity: {old_thresholds.cosine_similarity_threshold:.3f} → {best_cos_thresh:.3f}")
        logger.info(f"  PCA residual: {old_thresholds.pca_residual_threshold:.2f} → {best_pca_thresh:.2f}")

    def _find_optimal_threshold(
        self,
        safe_values: np.ndarray,
        attack_values: np.ndarray,
        higher_is_attack: bool,
        target_fpr: float = 0.05,
        target_tpr: float = 0.95
    ) -> float:
        """
        Find optimal threshold that balances FPR and TPR.

        Strategy: Find threshold closest to achieving target FPR while maximizing TPR.
        """
        # Combine all values and sort
        all_values = np.concatenate([safe_values, attack_values])
        all_values = np.sort(all_values)

        # Try thresholds at different percentiles
        percentiles = np.linspace(1, 99, 99)
        candidate_thresholds = np.percentile(all_values, percentiles)

        best_threshold = None
        best_score = -np.inf

        for thresh in candidate_thresholds:
            if higher_is_attack:
                # Attack = value > threshold
                fp = np.sum(safe_values > thresh)
                tp = np.sum(attack_values > thresh)
            else:
                # Attack = value < threshold
                fp = np.sum(safe_values < thresh)
                tp = np.sum(attack_values < thresh)

            fpr = fp / len(safe_values) if len(safe_values) > 0 else 0
            tpr = tp / len(attack_values) if len(attack_values) > 0 else 0

            # Score: prioritize achieving target FPR, then maximize TPR
            # Penalize heavily if FPR > target
            if fpr <= target_fpr:
                score = tpr  # Good FPR, maximize TPR
            else:
                score = -10 * (fpr - target_fpr)  # Penalize high FPR

            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_threshold if best_threshold is not None else np.median(all_values)

    def score_prompt(
        self,
        prompt: str
    ) -> AggregatedScores:
        """
        Score prompt for anomalies using multi-layer analysis.

        This is the main detection interface. It:
        1. Extracts embeddings from all selected layers
        2. Computes metrics per layer
        3. Aggregates across layers
        4. Returns comprehensive scores with anomaly flags

        Args:
            prompt: Input text to analyze

        Returns:
            AggregatedScores with detection results
        """
        if not self.layer_baselines:
            raise ValueError("Must call train_baseline() first")
        if not self.thresholds:
            raise ValueError("Thresholds not computed. Call train_baseline() with auto_tune=True")

        # Extract embeddings from all layers
        layer_embeddings = self.extract_multilayer_embeddings(prompt)

        # Compute per-layer scores
        layer_scores_list = []
        anomalous_layer_count = 0

        for layer_idx, embedding in layer_embeddings.items():
            baseline = self.layer_baselines[layer_idx]
            pca = self.layer_pcas[layer_idx]

            # Mahalanobis distance
            mahal = compute_mahalanobis_distance(
                embedding, baseline["mean"], baseline["cov_inv"]
            )

            # Cosine similarity
            cos_sim = compute_cosine_similarity(embedding, baseline["mean"])

            # PCA residual
            pca_proj = pca.transform([embedding])[0][:self.pca_reconstruction_components]
            pca_recon = pca.inverse_transform(
                np.pad(pca_proj, (0, pca.n_components_ - len(pca_proj)))
            )
            pca_residual = np.linalg.norm(embedding - pca_recon)

            # Top PCA projection
            pca_projs = compute_pca_projections(embedding, pca.components_)
            top_pca_proj = np.abs(pca_projs).max()

            # Check if this layer is anomalous
            is_layer_anomalous = (
                mahal > self.thresholds.mahalanobis_threshold or
                cos_sim < self.thresholds.cosine_similarity_threshold or
                pca_residual > self.thresholds.pca_residual_threshold
            )

            if is_layer_anomalous:
                anomalous_layer_count += 1

            layer_scores_list.append(LayerScores(
                layer_idx=layer_idx,
                mahalanobis_distance=mahal,
                cosine_similarity=cos_sim,
                pca_residual=pca_residual,
                top_pca_projection=top_pca_proj
            ))

        # Aggregate scores
        max_mahal = max(s.mahalanobis_distance for s in layer_scores_list)
        min_cos_sim = min(s.cosine_similarity for s in layer_scores_list)
        max_pca_residual = max(s.pca_residual for s in layer_scores_list)
        mean_mahal = np.mean([s.mahalanobis_distance for s in layer_scores_list])
        mean_cos_sim = np.mean([s.cosine_similarity for s in layer_scores_list])

        # Determine if anomalous based on detection mode
        is_anomalous = False
        anomaly_reasons = []

        if self.detection_mode == "voting":
            # Require multiple layers to agree
            if anomalous_layer_count >= self.min_anomalous_layers:
                is_anomalous = True
                anomaly_reasons.append(
                    f"{anomalous_layer_count}/{len(self.layer_indices)} layers flagged anomaly"
                )
        elif self.detection_mode == "max":
            # Use maximum metric across layers
            if max_mahal > self.thresholds.mahalanobis_threshold:
                is_anomalous = True
                anomaly_reasons.append(
                    f"Max Mahalanobis: {max_mahal:.2f} > {self.thresholds.mahalanobis_threshold:.2f}"
                )
            if min_cos_sim < self.thresholds.cosine_similarity_threshold:
                is_anomalous = True
                anomaly_reasons.append(
                    f"Min cosine similarity: {min_cos_sim:.3f} < {self.thresholds.cosine_similarity_threshold:.3f}"
                )
            if max_pca_residual > self.thresholds.pca_residual_threshold:
                is_anomalous = True
                anomaly_reasons.append(
                    f"Max PCA residual: {max_pca_residual:.2f} > {self.thresholds.pca_residual_threshold:.2f}"
                )
        elif self.detection_mode == "mean":
            # Use mean metric across layers
            if mean_mahal > self.thresholds.mahalanobis_threshold:
                is_anomalous = True
                anomaly_reasons.append(
                    f"Mean Mahalanobis: {mean_mahal:.2f} > {self.thresholds.mahalanobis_threshold:.2f}"
                )

        # Check for extreme outlier (mitigation threshold)
        exceeds_extreme = max_mahal > self.thresholds.extreme_mahalanobis_threshold

        # Compute confidence score (0-1)
        # Higher = more confident it's an attack
        confidence = self._compute_confidence_score(
            max_mahal, min_cos_sim, max_pca_residual, anomalous_layer_count
        )

        return AggregatedScores(
            max_mahalanobis=max_mahal,
            min_cosine_similarity=min_cos_sim,
            max_pca_residual=max_pca_residual,
            mean_mahalanobis=mean_mahal,
            mean_cosine_similarity=mean_cos_sim,
            is_anomalous=is_anomalous,
            confidence_score=confidence,
            anomaly_reasons=anomaly_reasons,
            exceeds_extreme_threshold=exceeds_extreme,
            layer_scores=layer_scores_list,
            num_anomalous_layers=anomalous_layer_count
        )

    def _compute_confidence_score(
        self,
        max_mahal: float,
        min_cos_sim: float,
        max_pca_residual: float,
        anomalous_layer_count: int
    ) -> float:
        """
        Compute confidence score for detection (0-1).

        Higher score = more confident it's an attack.
        """
        # Normalize each metric to 0-1 range
        mahal_score = min(
            max_mahal / (self.thresholds.extreme_mahalanobis_threshold + 1e-9),
            1.0
        )
        cos_sim_score = max(
            (self.thresholds.cosine_similarity_threshold - min_cos_sim) /
            (self.thresholds.cosine_similarity_threshold + 1e-9),
            0.0
        )
        pca_score = min(
            max_pca_residual / (self.thresholds.pca_residual_threshold + 1e-9),
            1.0
        )
        layer_vote_score = anomalous_layer_count / len(self.layer_indices)

        # Weighted combination
        confidence = (
            0.3 * mahal_score +
            0.2 * cos_sim_score +
            0.2 * pca_score +
            0.3 * layer_vote_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def generate_safe(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        apply_mitigation: bool = False
    ) -> Tuple[str, Dict]:
        """
        Generate text with optional selective mitigation.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            apply_mitigation: Whether to apply mitigation for extreme outliers

        Returns:
            Tuple of (generated_text, metrics_dict)
        """
        # Score the prompt
        scores = self.score_prompt(prompt)

        # Determine if mitigation should be applied
        should_mitigate = (
            apply_mitigation and
            scores.exceeds_extreme_threshold and
            len(self.attack_centroids) > 0
        )

        if should_mitigate:
            logger.info(
                f"Applying gentle mitigation (alpha={self.mitigation_alpha}) - "
                f"Extreme outlier detected (max Mahalanobis: {scores.max_mahalanobis:.2f})"
            )

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)
        input_length = input_ids.shape[1]

        with torch.no_grad():
            if should_mitigate:
                # Apply mitigation to selected layers
                # This requires modifying the forward pass, which is model-specific
                # For now, use simple embedding-level mitigation
                embedding_layer = self.model.get_input_embeddings()
                embeds = embedding_layer(input_ids)

                # Apply centroid-based projection at embedding level
                # (In a full implementation, you'd hook into specific layers)
                if 0 in self.attack_centroids:
                    centroid = torch.tensor(
                        self.attack_centroids[0],
                        dtype=torch.float32,
                        device=self.device
                    ).view(-1, 1)

                    embeds_flat = embeds.view(-1, embeds.shape[-1])
                    projection_magnitude = embeds_flat @ centroid
                    projection = projection_magnitude * centroid.T
                    embeds_flat = embeds_flat - self.mitigation_alpha * projection
                    embeds = embeds_flat.view(embeds.shape)

                outputs = self.model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            else:
                # Generate normally
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated_ids = outputs[:, input_length:]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            response = response.strip()

        # Compile metrics
        metrics = {
            "aggregated_scores": asdict(scores),
            "mitigation_applied": should_mitigate,
            "output_text": response
        }

        # Log if enabled
        if self.log_to_file:
            self._log_metrics(prompt, metrics)

        return response, metrics

    def _log_metrics(self, prompt: str, metrics: Dict):
        """Log metrics to file in JSONL format."""
        log_entry = {
            "prompt": prompt,
            **metrics
        }

        # Convert numpy types to Python native types
        log_entry = self._convert_to_native(log_entry)

        with open(self.log_file_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _convert_to_native(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_to_native(obj.tolist())
        elif isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._convert_to_native(asdict(obj))
        return obj

    def evaluate_dataset(
        self,
        prompts: List[str],
        labels: List[int],  # 0 = safe, 1 = attack
        apply_mitigation: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate detection performance on a labeled dataset.

        Args:
            prompts: List of prompts to evaluate
            labels: List of labels (0=safe, 1=attack)
            apply_mitigation: Whether to apply mitigation

        Returns:
            Dictionary with evaluation metrics:
            - detection_rate: True positive rate (attacks detected)
            - false_positive_rate: False positives / total safe
            - precision, recall, f1_score
            - per_prompt_results: List of per-prompt scores
        """
        logger.info(f"Evaluating {len(prompts)} prompts...")

        results = []
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        all_mahal_dists = []
        all_cos_sims = []
        all_pca_residuals = []
        all_confidence_scores = []

        for prompt, label in tqdm(zip(prompts, labels), total=len(prompts), desc="Evaluating"):
            scores = self.score_prompt(prompt)

            # Classification
            predicted = 1 if scores.is_anomalous else 0

            if label == 1 and predicted == 1:
                true_positives += 1
            elif label == 0 and predicted == 1:
                false_positives += 1
            elif label == 0 and predicted == 0:
                true_negatives += 1
            elif label == 1 and predicted == 0:
                false_negatives += 1

            # Collect metrics
            all_mahal_dists.append(scores.max_mahalanobis)
            all_cos_sims.append(scores.min_cosine_similarity)
            all_pca_residuals.append(scores.max_pca_residual)
            all_confidence_scores.append(scores.confidence_score)

            results.append({
                "prompt": prompt,
                "label": label,
                "predicted": predicted,
                "scores": asdict(scores)
            })

        # Compute metrics
        total_safe = labels.count(0)
        total_attack = labels.count(1)

        detection_rate = true_positives / total_attack if total_attack > 0 else 0.0
        false_positive_rate = false_positives / total_safe if total_safe > 0 else 0.0

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = detection_rate
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        accuracy = (true_positives + true_negatives) / len(prompts)

        evaluation_results = {
            "summary": {
                "total_prompts": len(prompts),
                "total_safe": total_safe,
                "total_attack": total_attack,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "detection_rate": detection_rate,
                "false_positive_rate": false_positive_rate,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": accuracy
            },
            "metric_distributions": {
                "mahalanobis": {
                    "mean": float(np.mean(all_mahal_dists)),
                    "std": float(np.std(all_mahal_dists)),
                    "min": float(np.min(all_mahal_dists)),
                    "max": float(np.max(all_mahal_dists))
                },
                "cosine_similarity": {
                    "mean": float(np.mean(all_cos_sims)),
                    "std": float(np.std(all_cos_sims)),
                    "min": float(np.min(all_cos_sims)),
                    "max": float(np.max(all_cos_sims))
                },
                "pca_residual": {
                    "mean": float(np.mean(all_pca_residuals)),
                    "std": float(np.std(all_pca_residuals)),
                    "min": float(np.min(all_pca_residuals)),
                    "max": float(np.max(all_pca_residuals))
                },
                "confidence_score": {
                    "mean": float(np.mean(all_confidence_scores)),
                    "std": float(np.std(all_confidence_scores)),
                    "min": float(np.min(all_confidence_scores)),
                    "max": float(np.max(all_confidence_scores))
                }
            },
            "per_prompt_results": results
        }

        logger.info("Evaluation complete:")
        logger.info(f"  Detection rate: {detection_rate:.2%}")
        logger.info(f"  False positive rate: {false_positive_rate:.2%}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall: {recall:.2%}")
        logger.info(f"  F1 score: {f1_score:.2%}")
        logger.info(f"  Accuracy: {accuracy:.2%}")

        return evaluation_results

    def save_model(self, path: str):
        """Save baseline statistics and configuration to file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        model_data = {
            "layer_indices": self.layer_indices,
            "thresholds": asdict(self.thresholds) if self.thresholds else None,
            "layer_baselines": {
                str(layer_idx): {
                    "mean": baseline["mean"].tolist(),
                    "cov": baseline["cov"].tolist(),
                    "cov_inv": baseline["cov_inv"].tolist()
                }
                for layer_idx, baseline in self.layer_baselines.items()
            },
            "layer_pcas": {
                str(layer_idx): {
                    "components": pca.components_.tolist(),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "mean": pca.mean_.tolist()
                }
                for layer_idx, pca in self.layer_pcas.items()
            },
            "attack_centroids": {
                str(layer_idx): centroid.tolist()
                for layer_idx, centroid in self.attack_centroids.items()
            },
            "config": {
                "detection_mode": self.detection_mode,
                "min_anomalous_layers": self.min_anomalous_layers,
                "mitigation_alpha": self.mitigation_alpha,
                "n_pca_components": self.n_pca_components,
                "pca_reconstruction_components": self.pca_reconstruction_components
            }
        }

        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load baseline statistics and configuration from file."""
        with open(path, 'r') as f:
            model_data = json.load(f)

        # Load thresholds
        if model_data["thresholds"]:
            self.thresholds = ThresholdConfig(**model_data["thresholds"])

        # Load layer baselines
        self.layer_baselines = {}
        for layer_idx_str, baseline_data in model_data["layer_baselines"].items():
            layer_idx = int(layer_idx_str)
            self.layer_baselines[layer_idx] = {
                "mean": np.array(baseline_data["mean"]),
                "cov": np.array(baseline_data["cov"]),
                "cov_inv": np.array(baseline_data["cov_inv"])
            }

        # Load PCAs
        self.layer_pcas = {}
        for layer_idx_str, pca_data in model_data["layer_pcas"].items():
            layer_idx = int(layer_idx_str)
            pca = PCA(n_components=len(pca_data["components"]))
            pca.components_ = np.array(pca_data["components"])
            pca.explained_variance_ratio_ = np.array(pca_data["explained_variance_ratio"])
            pca.mean_ = np.array(pca_data["mean"])
            self.layer_pcas[layer_idx] = pca

        # Load attack centroids
        self.attack_centroids = {
            int(layer_idx_str): np.array(centroid)
            for layer_idx_str, centroid in model_data["attack_centroids"].items()
        }

        logger.info(f"Model loaded from {path}")
        logger.info(f"Loaded baselines for layers: {list(self.layer_baselines.keys())}")
