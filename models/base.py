# Abstract Base Recommender - Clean Interface Design
# Follows SOLID principles with extensible architecture

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np
from scipy import sparse


class BaseRecommender(ABC):
    """Abstract base class for all recommendation models.
    
    Provides consistent interface for:
    - Training (fit)
    - Prediction (recommend)
    - Explanation (explain)
    - Evaluation (predict_scores)
    
    All concrete implementations must override abstract methods.
    """
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._is_fitted = False
        self._n_users: int = 0
        self._n_items: int = 0
        self._user_ids: Optional[np.ndarray] = None
        self._item_ids: Optional[np.ndarray] = None
        
    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted
    
    @abstractmethod
    def fit(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame = None,
        items: pd.DataFrame = None
    ) -> "BaseRecommender":
        """Train the recommendation model.
        
        Args:
            interactions: DataFrame with user_id, item_id, and interaction data
            users: Optional user metadata DataFrame
            items: Optional item metadata DataFrame
            
        Returns:
            self for method chaining
        """
        pass
    
    @abstractmethod
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations for a user.
        
        Args:
            user_id: Target user ID
            n: Number of recommendations
            exclude_seen: Whether to exclude items user has interacted with
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        pass
    
    @abstractmethod
    def explain(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """Explain why an item is recommended for a user.
        
        Args:
            user_id: Target user
            item_id: Recommended item
            
        Returns:
            Dictionary containing explanation details
        """
        pass
    
    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict score for a single user-item pair.
        
        Override for optimized implementation.
        """
        recs = self.recommend(user_id, n=self._n_items, exclude_seen=False)
        for rec_item_id, score in recs:
            if rec_item_id == item_id:
                return score
        return 0.0
    
    def batch_recommend(
        self,
        user_ids: List[int],
        n: int = 10,
        exclude_seen: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            exclude_seen: Whether to exclude seen items
            
        Returns:
            Dictionary mapping user_id to recommendation list
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, n, exclude_seen)
        return results
    
    def get_similar_users(
        self,
        user_id: int,
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """Get most similar users (override for CF models)."""
        return []
    
    def get_similar_items(
        self,
        item_id: int,
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """Get most similar items (override for CF models)."""
        return []
    
    def _validate_fitted(self) -> None:
        """Raise error if model not fitted."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before use")
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}(name='{self.name}', {status})"
