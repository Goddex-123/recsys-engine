# Core Configuration - Scalable Recommendation System
# Designed for horizontal scaling with configurable parameters

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
import os


class Environment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelType(Enum):
    """Available recommendation model types."""
    POPULARITY = "popularity"
    USER_CF = "user_collaborative"
    ITEM_CF = "item_collaborative"
    SVD = "matrix_factorization"
    HYBRID = "hybrid"


@dataclass
class DataConfig:
    """Data generation and storage configuration.
    
    Designed for production scale:
    - Users: 10K (typical mid-size platform)
    - Items: 5K (catalog size)
    - Interactions: 500K (realistic engagement rate ~10 per user)
    """
    n_users: int = 10_000
    n_items: int = 5_000
    n_interactions: int = 500_000
    
    # Interaction type weights (must sum to 1.0)
    interaction_weights: Dict[str, float] = field(default_factory=lambda: {
        "view": 0.50,      # Most common
        "click": 0.25,     # Medium engagement
        "like": 0.15,      # Active engagement
        "purchase": 0.10   # Conversion
    })
    
    # Power-law distribution parameters (Pareto principle)
    popularity_alpha: float = 1.5  # Item popularity skew
    activity_alpha: float = 1.2    # User activity skew
    
    # Temporal configuration
    history_days: int = 365
    time_decay_halflife: int = 30  # Days
    
    # Storage paths
    data_dir: str = "generated_data"
    

@dataclass
class ModelConfig:
    """Model training and inference configuration."""
    
    # Collaborative filtering
    similarity_metric: str = "cosine"  # cosine, pearson, jaccard
    min_neighbors: int = 5
    max_neighbors: int = 50
    min_common_items: int = 3  # For user similarity
    
    # Matrix factorization (SVD)
    n_factors: int = 100
    n_epochs: int = 20
    learning_rate: float = 0.005
    regularization: float = 0.02
    
    # Hybrid weights
    collaborative_weight: float = 0.6
    content_weight: float = 0.3
    popularity_weight: float = 0.1
    
    # Cold-start thresholds
    cold_start_user_threshold: int = 5   # Min interactions for warm user
    cold_start_item_threshold: int = 10  # Min interactions for warm item
    
    # Recommendation settings
    default_top_n: int = 10
    diversity_weight: float = 0.2  # For re-ranking


@dataclass
class EvaluationConfig:
    """Offline evaluation configuration."""
    test_ratio: float = 0.2
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    n_folds: int = 5  # For cross-validation
    random_seed: int = 42


@dataclass
class DashboardConfig:
    """Streamlit dashboard configuration."""
    page_title: str = "RecSys Engine"
    page_icon: str = "🎯"
    layout: str = "wide"
    theme_primary_color: str = "#6366F1"  # Indigo
    theme_secondary_color: str = "#8B5CF6"  # Violet
    theme_background: str = "#0F172A"  # Slate 900
    theme_card_bg: str = "#1E293B"  # Slate 800


@dataclass
class SystemConfig:
    """Master configuration aggregating all sub-configs."""
    environment: Environment = Environment.DEVELOPMENT
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Scalability settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    enable_async_updates: bool = False  # For production
    batch_size: int = 1000
    
    @classmethod
    def for_environment(cls, env: str) -> "SystemConfig":
        """Factory method for environment-specific configs."""
        config = cls(environment=Environment(env))
        
        if config.environment == Environment.PRODUCTION:
            config.data.n_users = 100_000
            config.data.n_items = 50_000
            config.data.n_interactions = 10_000_000
            config.enable_async_updates = True
            config.batch_size = 10_000
            
        return config


# Global config instance
config = SystemConfig()
