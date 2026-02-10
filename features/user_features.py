# User Feature Extraction
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


class UserFeatureExtractor:
    """Extract behavioral features for users.
    
    Features:
    - Activity metrics (interaction count, recency)
    - Category affinity vectors
    - Engagement patterns (time-of-day, frequency)
    """
    
    def __init__(self):
        self._user_features: Dict[int, Dict] = {}
        self._category_list: List[str] = []
    
    def fit(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame = None
    ) -> "UserFeatureExtractor":
        """Extract features from interaction data."""
        
        # Build category list
        if items_df is not None:
            self._category_list = sorted(items_df['category'].unique().tolist())
            item_to_cat = dict(zip(items_df['item_id'], items_df['category']))
        else:
            self._category_list = []
            item_to_cat = {}
        
        # Interaction type weights
        type_weights = {'view': 1.0, 'click': 2.0, 'like': 4.0, 'purchase': 5.0}
        
        now = datetime.now()
        
        for user_id in interactions_df['user_id'].unique():
            user_ints = interactions_df[interactions_df['user_id'] == user_id]
            
            # Basic metrics
            n_interactions = len(user_ints)
            
            # Recency
            try:
                timestamps = pd.to_datetime(user_ints['timestamp'])
                most_recent = timestamps.max()
                days_since_active = (now - most_recent).days if pd.notna(most_recent) else 365
            except:
                days_since_active = 365
            
            # Weighted interaction score
            weighted_score = sum(
                type_weights.get(row['interaction_type'], 1.0)
                for _, row in user_ints.iterrows()
            )
            
            # Category affinity
            cat_counts = defaultdict(float)
            for _, row in user_ints.iterrows():
                item_id = row['item_id']
                if item_id in item_to_cat:
                    cat = item_to_cat[item_id]
                    weight = type_weights.get(row['interaction_type'], 1.0)
                    cat_counts[cat] += weight
            
            # Normalize to vector
            total_cat = sum(cat_counts.values()) or 1
            cat_vector = np.array([
                cat_counts.get(cat, 0) / total_cat
                for cat in self._category_list
            ])
            
            # Store features
            self._user_features[user_id] = {
                'n_interactions': n_interactions,
                'days_since_active': days_since_active,
                'weighted_score': weighted_score,
                'engagement_rate': weighted_score / max(1, n_interactions),
                'category_affinity': cat_vector,
                'top_categories': sorted(
                    cat_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
        
        return self
    
    def get_features(self, user_id: int) -> Optional[Dict]:
        """Get extracted features for a user."""
        return self._user_features.get(user_id)
    
    def get_activity_score(self, user_id: int) -> float:
        """Get normalized activity score."""
        features = self._user_features.get(user_id)
        if not features:
            return 0.0
        
        # Combine metrics
        recency_score = max(0, 1 - features['days_since_active'] / 365)
        engagement = min(1, features['weighted_score'] / 100)
        
        return 0.6 * engagement + 0.4 * recency_score
    
    def is_cold_start(self, user_id: int, threshold: int = 5) -> bool:
        """Check if user is cold-start."""
        features = self._user_features.get(user_id)
        if not features:
            return True
        return features['n_interactions'] < threshold
