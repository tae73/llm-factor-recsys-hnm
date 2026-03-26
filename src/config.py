"""Global configuration NamedTuples for the H&M LLM-Factor RecSys project.

All modules import config objects from here to ensure consistent interfaces.
"""

from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------


class DataPaths(NamedTuple):
    """Paths to raw and processed data directories."""

    raw_dir: Path
    processed_dir: Path
    articles_csv: str = "articles.csv"
    customers_csv: str = "customers.csv"
    transactions_csv: str = "transactions_train.csv"


# ---------------------------------------------------------------------------
# Split / Filter Configuration
# ---------------------------------------------------------------------------


class SplitConfig(NamedTuple):
    """Temporal split boundaries for train/val/test."""

    train_end: str = "2020-06-30"
    val_start: str = "2020-07-01"
    val_end: str = "2020-08-31"
    test_start: str = "2020-09-01"
    test_end: str = "2020-09-07"


class FilterConfig(NamedTuple):
    """Customer activity filtering thresholds."""

    active_min: int = 5  # >= active_min purchases = active user
    sparse_min: int = 1  # 1 ~ active_min-1 purchases = sparse user


# ---------------------------------------------------------------------------
# Evaluation Configuration
# ---------------------------------------------------------------------------


class EvalConfig(NamedTuple):
    """Evaluation metric settings."""

    k: int = 12
    metrics: tuple[str, ...] = ("map", "hr", "ndcg", "mrr")


# ---------------------------------------------------------------------------
# Baseline Configuration
# ---------------------------------------------------------------------------


class BaselineConfig(NamedTuple):
    """Hyperparameters for baseline models."""

    # ALS (UserKNN)
    als_factors: int = 128
    als_regularization: float = 0.01
    als_iterations: int = 15

    # BPR-MF
    bpr_factors: int = 128
    bpr_learning_rate: float = 0.01
    bpr_iterations: int = 100

    # Popularity
    popularity_window_days: int = 7


# ---------------------------------------------------------------------------
# Result Objects
# ---------------------------------------------------------------------------


class PreprocessResult(NamedTuple):
    """Summary of preprocessing output."""

    articles_path: Path
    customers_path: Path
    transactions_path: Path
    n_articles: int
    n_customers: int
    n_transactions: int


class SplitResult(NamedTuple):
    """Summary of temporal split output."""

    train_path: Path
    val_path: Path
    test_path: Path
    n_train: int
    n_val: int
    n_test: int
    n_active_users: int
    n_sparse_users: int
    n_cold_start_users_val: int
    n_cold_start_items_val: int


class EvalResult(NamedTuple):
    """Evaluation metric results."""

    map_at_k: float
    hr_at_k: float
    ndcg_at_k: float
    mrr: float


# ---------------------------------------------------------------------------
# Interaction Data
# ---------------------------------------------------------------------------


class InteractionData(NamedTuple):
    """Sparse interaction matrix with index mappings."""

    matrix: csr_matrix  # (n_users, n_items) CSR matrix
    user_to_idx: dict[str, int]
    idx_to_user: dict[int, str]
    item_to_idx: dict[str, int]
    idx_to_item: dict[int, str]


# ---------------------------------------------------------------------------
# Knowledge Extraction Configuration
# ---------------------------------------------------------------------------


class ExtractionConfig(NamedTuple):
    """LLM knowledge extraction settings."""

    model: str = "gpt-4.1-nano"
    use_batch_api: bool = False  # True for full batch, False for pilot
    max_concurrent: int = 5  # 실시간 API 동시 요청 수 (TPM 200K 기준)
    checkpoint_interval: int = 100  # 체크포인트 저장 간격 (product_code 수)
    max_cost_usd: float = 15.0  # 비용 가드 (Batch API 기준)
    max_retries: int = 6
    timeout_seconds: float = 30.0
    image_max_size: int = 512  # 이미지 리사이즈 (픽셀)
    pilot_size: int = 500  # 파일럿 추출 수
    tpm_limit: int = 200_000  # tokens-per-minute limit for real-time API
    batch_max_bytes: int = 150_000_000  # 청크 분할 기준 (150MB, 200MB 한도 대비 안전 마진)
    batch_max_requests: int = 500  # 청크당 최대 요청 수 (500 × ~3K tokens ≈ 1.5M < 2M enqueued limit)


class ExtractionResult(NamedTuple):
    """Extraction run summary."""

    output_path: Path
    n_products: int  # 고유 product_code 수
    n_articles: int  # 전체 article 수 (변형 포함)
    n_api_calls: int  # 실제 API 호출 수
    n_cache_hits: int  # 체크포인트 캐시 히트 수
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    coverage: dict[str, float]  # 속성별 non-null 비율


# ---------------------------------------------------------------------------
# User Reasoning Knowledge Configuration
# ---------------------------------------------------------------------------


class ReasoningConfig(NamedTuple):
    """User reasoning knowledge extraction settings (L2+L3)."""

    model: str = "gpt-4.1-nano"
    use_batch_api: bool = False
    max_concurrent: int = 5
    checkpoint_interval: int = 500
    max_cost_usd: float = 120.0
    max_retries: int = 6
    timeout_seconds: float = 30.0
    tpm_limit: int = 200_000
    batch_max_bytes: int = 150_000_000
    batch_max_requests: int = 1_000
    pilot_size: int = 200
    min_purchases: int = 5  # Active user threshold
    recent_items_limit: int = 20  # Recent items for L2 input
    l1_time_weight_halflife_days: int = 90  # Exponential decay halflife


class ReasoningResult(NamedTuple):
    """User reasoning knowledge extraction run summary."""

    output_path: Path
    n_active_users: int
    n_sparse_users: int
    n_api_calls: int
    n_cache_hits: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    avg_reasoning_text_tokens: float


# ---------------------------------------------------------------------------
# Feature Engineering Configuration
# ---------------------------------------------------------------------------


class FeatureConfig(NamedTuple):
    """Feature engineering settings."""

    neg_sample_ratio: int = 4
    reference_date: str = "2020-06-30"
    age_bins: tuple[int, ...] = (0, 18, 25, 35, 45, 55, 65, 100)
    age_labels: tuple[str, ...] = (
        "teen",
        "young_adult",
        "adult",
        "middle_age",
        "senior",
        "elderly",
        "unknown",
    )
    random_seed: int = 42
    chunk_size: int = 100_000


class FeatureResult(NamedTuple):
    """Feature engineering output summary."""

    output_dir: Path
    n_users: int
    n_items: int
    n_train_pairs: int
    n_user_num_features: int
    n_user_cat_features: int
    n_item_num_features: int
    n_item_cat_features: int
    user_cat_vocab_sizes: dict[str, int]
    item_cat_vocab_sizes: dict[str, int]


# ---------------------------------------------------------------------------
# DeepFM Configuration
# ---------------------------------------------------------------------------


class DeepFMConfig(NamedTuple):
    """DeepFM model hyperparameters."""

    d_embed: int = 16
    dnn_hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True


class DCNv2Config(NamedTuple):
    """DCN-v2 (Deep & Cross Network v2) model hyperparameters."""

    d_embed: int = 16
    n_cross_layers: int = 3  # cross layer 수
    n_experts: int = 4  # MoE experts per cross layer
    d_low_rank: int = 64  # low-rank dim per expert
    dnn_hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True


class LightGCNConfig(NamedTuple):
    """LightGCN model hyperparameters."""

    d_embed: int = 64
    n_layers: int = 3  # propagation layers
    dropout_rate: float = 0.0  # edge dropout
    l2_reg: float = 1e-4  # L2 on initial embeddings


class SequenceConfig(NamedTuple):
    """Sequential feature settings."""

    max_seq_len: int = 50
    random_seed: int = 42


class DINConfig(NamedTuple):
    """DIN (Deep Interest Network) model hyperparameters."""

    d_embed: int = 16
    attention_hidden_dims: tuple[int, ...] = (64, 32)
    dnn_hidden_dims: tuple[int, ...] = (256, 128, 64)
    dropout_rate: float = 0.1
    use_batch_norm: bool = True


class SASRecConfig(NamedTuple):
    """SASRec (Self-Attentive Sequential Recommendation) model hyperparameters."""

    d_embed: int = 64
    n_heads: int = 2
    n_blocks: int = 2
    max_seq_len: int = 50
    dropout_rate: float = 0.2


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


class TrainConfig(NamedTuple):
    """Training loop settings."""

    learning_rate: float = 1e-3
    batch_size: int = 2048
    max_epochs: int = 50
    patience: int = 3
    val_every_n_steps: int = 5000
    val_sample_users: int = 1000
    random_seed: int = 42
    log_every_n_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "llm-factor-recsys-hnm"
    num_workers: int = 4  # Grain multiprocess workers (0 = same process)
    prefetch_buffer_size: int = 2  # Batches to prefetch per worker


class TrainResult(NamedTuple):
    """Training run summary."""

    model_dir: Path
    best_epoch: int
    best_val_map_at_12: float
    best_val_hr_at_12: float
    best_val_ndcg_at_12: float
    best_val_mrr: float
    total_train_steps: int
    total_train_time_seconds: float
    n_devices: int


# ---------------------------------------------------------------------------
# Embedding Configuration
# ---------------------------------------------------------------------------


class EmbeddingConfig(NamedTuple):
    """BGE embedding computation settings (shared across segmentation & KAR)."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = 256
    max_seq_length: int = 512
    device: str = "mps"


# ---------------------------------------------------------------------------
# Segmentation Configuration
# ---------------------------------------------------------------------------


class SegmentationConfig(NamedTuple):
    """Customer/Item segmentation settings."""

    customer_k_range: tuple[int, ...] = (4, 6, 8, 10, 12, 15)
    customer_method: str = "kmeans"
    pca_variance_threshold: float = 0.95
    umap_n_components: int = 2
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_cluster_n_components: int = 5
    hdbscan_min_cluster_size: int = 500
    hdbscan_min_samples: int = 50
    product_k_range: tuple[int, ...] = (10, 15, 20, 25, 30)
    product_method: str = "kmeans"
    subsample_size: int = 50_000
    random_seed: int = 42


class SegmentationResult(NamedTuple):
    """Segmentation output summary."""

    output_dir: Path
    n_item_embeddings: int
    n_user_embeddings: int
    customer_segment_counts: dict[str, int]
    product_n_clusters: int
    product_silhouette: float
    topic_n_topics: int
    topic_outlier_ratio: float
    cross_layer_ari: dict[str, float]


# ---------------------------------------------------------------------------
# KAR (Knowledge-Augmented Recommendation) Configuration
# ---------------------------------------------------------------------------


class ExpertConfig(NamedTuple):
    """Expert MLP hyperparameters (factual + reasoning share same architecture)."""

    d_enc: int = 768  # BGE input dimension
    d_hidden: int = 256  # MLP hidden dimension
    d_rec: int = 64  # Expert output dimension
    n_layers: int = 2
    dropout_rate: float = 0.1


class GatingConfig(NamedTuple):
    """Gating network variant selection."""

    variant: str = "g2"  # g1|g2|g3|g4
    d_context: int = 0  # G3: user demographic context dimension


class FusionConfig(NamedTuple):
    """Fusion strategy variant selection."""

    variant: str = "f2"  # f1|f2|f3|f4
    alpha_init: float = 0.1  # F2: initial scaling factor
    n_heads: int = 4  # F4: cross-attention heads


class KARConfig(NamedTuple):
    """Full KAR module configuration."""

    expert: ExpertConfig = ExpertConfig()
    gating: GatingConfig = GatingConfig()
    fusion: FusionConfig = FusionConfig()
    layer_combo: str = "L1+L2+L3"
    align_weight: float = 0.1
    diversity_weight: float = 0.01
    stage1_epochs: int = 20  # Backbone pre-train (BCE only, with early stopping)
    stage2_epochs: int = 5  # Expert adaptor (align+div, backbone frozen)
    stage3_epochs: int = 3  # End-to-end (BCE+align+div)
    stage3_lr_factor: float = 0.1  # LR multiplier for stage 3


# ---------------------------------------------------------------------------
# Re-Ranker Configuration (GBDT 2nd-stage)
# ---------------------------------------------------------------------------


class ReRankerConfig(NamedTuple):
    """LightGBM GBDT Re-Ranker hyperparameters."""

    top_k: int = 100  # Stage 1 candidate pool size
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_seed: int = 42


class ReRankerResult(NamedTuple):
    """Re-ranker training + evaluation summary."""

    output_dir: Path
    n_train_samples: int
    n_val_samples: int
    n_features: int
    best_iteration: int
    val_auc: float
    map_at_12: float
    hr_at_12: float
    ndcg_at_12: float
    mrr: float
    top_features: list[tuple[str, float]]  # (name, importance) top-20
