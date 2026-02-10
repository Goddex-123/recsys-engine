# Popularity-Based Recommender - Strong Baseline
# Non-personalized but effective for cold-start and comparison

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Popularity-based recommendation model.
    
    Recommends items based on global popularity metrics.
    Serves as essential baseline and cold-start fallback.
    
    Features:
    - Time-weighted popularity (recent interactions count more)
    - Category-specific popularity
    - Configurable decay function
    
    Complexity: O(n log n) for sorting ranked items
    """
    
    def __init__(
        self,
        time_decay_days: int = 30,
        use_time_decay: bool = True
    ):
        super().__init__(
            name="Popularity",
            description="Global popularity-based recommendations with time decay"
        )
        self.time_decay_days = time_decay_days
        self.use_time_decay = use_time_decay
        
        # Learned parameters
        self._item_scores: Dict[int, float] = {}
        self._category_scores: Dict[str, Dict[int, float]] = {}
        self._user_seen: Dict[int, set] = defaultdict(set)
        self._items_df: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame = None,
        items: pd.DataFrame = None
    ) -> "PopularityRecommender":
        """Compute popularity scores from interaction data."""
        
        self._n_items = interactions['item_id'].nunique()
        self._n_users = interactions['user_id'].nunique()
        self._items_df = items
        
        # Build user history for exclusion
        for _, row in interactions.iterrows():
            self._user_seen[row['user_id']].add(row['item_id'])
        
        # Interaction type weights
        type_weights = {
            'view': 1.0,
            'click': 2.0,
            'like': 4.0,
            'purchase': 5.0
        }
        
        # Calculate time-weighted scores
        now = datetime.now()
        item_scores = defaultdict(float)
        
        for _, row in interactions.iterrows():
            item_id = row['item_id']
            interaction_type = row['interaction_type']
            
            base_weight = type_weights.get(interaction_type, 1.0)
            
            # Apply time decay
            if self.use_time_decay:
                try:
                    if isinstance(row['timestamp'], str):
                        ts = datetime.fromisoformat(row['timestamp'])
                    else:
                        ts = row['timestamp']
                    days_ago = (now - ts).days
                    decay = 2 ** (-days_ago / self.time_decay_days)
                except:
                    decay = 0.5
            else:
                decay = 1.0
            
            item_scores[item_id] += base_weight * decay
        
        # Normalize scores to 0-1 range
        if item_scores:
            max_score = max(item_scores.values())
            self._item_scores = {
                k: v / max_score for k, v in item_scores.items()
            }
        
        # Category-specific popularity
        if items is not None:
            item_to_cat = dict(zip(items['item_id'], items['category']))
            cat_scores = defaultdict(lambda: defaultdict(float))
            
            for item_id, score in self._item_scores.items():
                if item_id in item_to_cat:
                    cat = item_to_cat[item_id]
                    cat_scores[cat][item_id] = score
            
            self._category_scores = dict(cat_scores)
        
        self._is_fitted = True
        return self
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        category: str = None
    ) -> List[Tuple[int, float]]:
        """Get top-N popular items."""
        self._validate_fitted()
        
        # Get scores (optionally filtered by category)
        if category and category in self._category_scores:
            scores = self._category_scores[category]
        else:
            scores = self._item_scores
        
        # Filter seen items
        if exclude_seen:
            seen = self._user_seen.get(user_id, set())
            scores = {k: v for k, v in scores.items() if k not in seen}
        
        # Sort and return top-N
        sorted_items = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_items[:n]
    
    def explain(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """Explain popularity-based recommendation."""
        self._validate_fitted()
        
        score = self._item_scores.get(item_id, 0.0)
        rank = sum(1 for s in self._item_scores.values() if s > score) + 1
        
        item_info = {}
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_info = item_row.iloc[0].to_dict()
        
        return {
            "model": self.name,
            "reason": f"Trending item (#{rank} overall)",
            "score": score,
            "rank": rank,
            "total_items": len(self._item_scores),
            "explanation_text": f"This is currently #{rank} in popularity with a trending score of {score:.2%}",
            "factors": [
                {"name": "Popularity Rank", "value": f"#{rank}"},
                {"name": "Trending Score", "value": f"{score:.1%}"},
                {"name": "Category", "value": item_info.get("category", "N/A")}
            ]
        }
    
    def get_trending(
        self,
        n: int = 10,
        category: str = None
    ) -> List[Tuple[int, float]]:
        """Get trending items without user context."""
        return self.recommend(user_id=-1, n=n, exclude_seen=False, category=category)
