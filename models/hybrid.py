# Hybrid Recommender - Ensemble Approach
# Combines collaborative, content, and popularity signals

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

from .base import BaseRecommender
from .popularity import PopularityRecommender
from .user_cf import UserBasedCF
from .item_cf import ItemBasedCF
from .svd import SVDRecommender


class HybridRecommender(BaseRecommender):
    """Hybrid recommendation model combining multiple strategies.
    
    Ensemble approach with:
    - Content-based filtering (category/tag similarity)
    - Collaborative filtering (SVD-based)
    - Popularity baseline (for cold-start)
    
    Features:
    - Weighted score fusion
    - Automatic cold-start handling
    - Configurable component weights
    
    Strategy:
    1. For warm users: Blend CF + content + popularity
    2. For cold users: Fall back to content + popularity
    3. For cold items: Use content similarity
    """
    
    def __init__(
        self,
        collaborative_weight: float = 0.5,
        content_weight: float = 0.3,
        popularity_weight: float = 0.2,
        cold_start_threshold: int = 5
    ):
        super().__init__(
            name="Hybrid",
            description="Ensemble combining collaborative, content, and popularity signals"
        )
        
        # Validate weights
        total = collaborative_weight + content_weight + popularity_weight
        self.collaborative_weight = collaborative_weight / total
        self.content_weight = content_weight / total
        self.popularity_weight = popularity_weight / total
        
        self.cold_start_threshold = cold_start_threshold
        
        # Component models
        self._svd: Optional[SVDRecommender] = None
        self._item_cf: Optional[ItemBasedCF] = None
        self._popularity: Optional[PopularityRecommender] = None
        
        # Content similarity
        self._item_content_vectors: Dict[int, np.ndarray] = {}
        self._category_to_idx: Dict[str, int] = {}
        self._tag_to_idx: Dict[str, int] = {}
        
        # User profiles
        self._user_interaction_counts: Dict[int, int] = {}
        self._user_category_prefs: Dict[int, Dict[str, float]] = {}
        
        self._items_df: Optional[pd.DataFrame] = None
        self._users_df: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame = None,
        items: pd.DataFrame = None
    ) -> "HybridRecommender":
        """Train all component models."""
        
        print(f"[TRAIN] Training {self.name} (ensemble)...")
        
        self._items_df = items
        self._users_df = users
        
        # Build user interaction counts
        user_counts = interactions.groupby('user_id').size()
        self._user_interaction_counts = user_counts.to_dict()
        
        # Build user category preferences
        if items is not None:
            item_to_cat = dict(zip(items['item_id'], items['category']))
            
            for user_id in interactions['user_id'].unique():
                user_interactions = interactions[interactions['user_id'] == user_id]
                cat_counts = defaultdict(float)
                
                type_weights = {'view': 1.0, 'click': 2.0, 'like': 4.0, 'purchase': 5.0}
                
                for _, row in user_interactions.iterrows():
                    item_id = row['item_id']
                    if item_id in item_to_cat:
                        cat = item_to_cat[item_id]
                        weight = type_weights.get(row['interaction_type'], 1.0)
                        cat_counts[cat] += weight
                
                # Normalize
                total = sum(cat_counts.values()) or 1
                self._user_category_prefs[user_id] = {
                    cat: count / total for cat, count in cat_counts.items()
                }
        
        # Build content vectors for items
        if items is not None:
            self._build_content_vectors(items)
        
        # Train component models
        print("   Training SVD component...")
        self._svd = SVDRecommender(n_factors=50)
        self._svd.fit(interactions, users, items)
        
        print("   Training Item-CF component...")
        self._item_cf = ItemBasedCF(k_neighbors=20)
        self._item_cf.fit(interactions, users, items)
        
        print("   Training Popularity component...")
        self._popularity = PopularityRecommender()
        self._popularity.fit(interactions, users, items)
        
        self._n_users = interactions['user_id'].nunique()
        self._n_items = interactions['item_id'].nunique()
        
        self._is_fitted = True
        print(f"   [OK] {self.name} trained with 3 components")
        
        return self
    
    def _build_content_vectors(self, items: pd.DataFrame) -> None:
        """Build content feature vectors for items."""
        
        # Collect all categories and tags
        categories = items['category'].unique().tolist()
        all_tags = set()
        
        for tags in items['tags']:
            if isinstance(tags, list):
                all_tags.update(tags)
            elif isinstance(tags, str):
                try:
                    import ast
                    tag_list = ast.literal_eval(tags)
                    all_tags.update(tag_list)
                except:
                    pass
        
        all_tags = sorted(list(all_tags))
        
        # Create index mappings
        self._category_to_idx = {cat: i for i, cat in enumerate(categories)}
        self._tag_to_idx = {tag: i + len(categories) for i, tag in enumerate(all_tags)}
        
        vector_size = len(categories) + len(all_tags)
        
        # Build vectors
        for _, row in items.iterrows():
            item_id = row['item_id']
            vector = np.zeros(vector_size)
            
            # Category (one-hot)
            if row['category'] in self._category_to_idx:
                vector[self._category_to_idx[row['category']]] = 1.0
            
            # Tags (multi-hot)
            tags = row['tags']
            if isinstance(tags, str):
                try:
                    import ast
                    tags = ast.literal_eval(tags)
                except:
                    tags = []
            
            if isinstance(tags, list):
                for tag in tags:
                    if tag in self._tag_to_idx:
                        vector[self._tag_to_idx[tag]] = 1.0
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            self._item_content_vectors[item_id] = vector
    
    def _content_similarity(self, item_id1: int, item_id2: int) -> float:
        """Compute content similarity between items."""
        vec1 = self._item_content_vectors.get(item_id1)
        vec2 = self._item_content_vectors.get(item_id2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return float(np.dot(vec1, vec2))
    
    def _get_content_recommendations(
        self,
        user_id: int,
        n: int,
        exclude_items: set
    ) -> List[Tuple[int, float]]:
        """Get content-based recommendations for user."""
        
        user_prefs = self._user_category_prefs.get(user_id, {})
        
        if not user_prefs:
            return []
        
        # Score items by category match
        item_scores = {}
        
        if self._items_df is not None:
            for _, row in self._items_df.iterrows():
                item_id = row['item_id']
                
                if item_id in exclude_items:
                    continue
                
                category = row['category']
                score = user_prefs.get(category, 0.0)
                
                # Add freshness bonus
                if 'avg_rating' in row:
                    score *= (0.5 + 0.5 * (row['avg_rating'] / 5.0))
                
                if score > 0:
                    item_scores[item_id] = score
        
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return sorted_items
    
    def _is_cold_start_user(self, user_id: int) -> bool:
        """Check if user is cold-start."""
        return self._user_interaction_counts.get(user_id, 0) < self.cold_start_threshold
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate hybrid recommendations."""
        self._validate_fitted()
        
        is_cold = self._is_cold_start_user(user_id)
        
        # Get recommendations from each component
        collab_recs = []
        content_recs = []
        popularity_recs = []
        
        # Seen items for exclusion
        seen_items = set()
        if exclude_seen:
            svd_matrix = self._svd._user_item_matrix
            user_idx = self._svd._user_idx_map.get(user_id)
            if user_idx is not None:
                seen_mask = svd_matrix[user_idx].toarray().flatten() > 0
                seen_items = {
                    self._svd._idx_item_map[i] 
                    for i in np.where(seen_mask)[0]
                }
        
        # Collaborative recommendations (skip for cold users)
        if not is_cold:
            # Blend SVD and Item-CF
            svd_recs = self._svd.recommend(user_id, n=n*2, exclude_seen=exclude_seen)
            itemcf_recs = self._item_cf.recommend(user_id, n=n*2, exclude_seen=exclude_seen)
            
            # Merge with equal weights
            collab_scores = defaultdict(float)
            for item_id, score in svd_recs:
                collab_scores[item_id] += score * 0.5
            for item_id, score in itemcf_recs:
                collab_scores[item_id] += score * 0.5
            
            collab_recs = sorted(collab_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Content recommendations
        content_recs = self._get_content_recommendations(
            user_id, n=n*2, exclude_items=seen_items
        )
        
        # Popularity recommendations
        popularity_recs = self._popularity.recommend(
            user_id, n=n*2, exclude_seen=exclude_seen
        )
        
        # Determine weights based on cold-start status
        if is_cold:
            weights = {
                'collab': 0.0,
                'content': 0.6,
                'popularity': 0.4
            }
        else:
            weights = {
                'collab': self.collaborative_weight,
                'content': self.content_weight,
                'popularity': self.popularity_weight
            }
        
        # Normalize scores and combine
        def normalize_scores(recs: List[Tuple[int, float]]) -> Dict[int, float]:
            if not recs:
                return {}
            max_score = max(s for _, s in recs) or 1
            return {item_id: score / max_score for item_id, score in recs}
        
        collab_norm = normalize_scores(collab_recs)
        content_norm = normalize_scores(content_recs)
        pop_norm = normalize_scores(popularity_recs)
        
        # Aggregate
        final_scores = defaultdict(float)
        all_items = set(collab_norm.keys()) | set(content_norm.keys()) | set(pop_norm.keys())
        
        for item_id in all_items:
            final_scores[item_id] = (
                weights['collab'] * collab_norm.get(item_id, 0) +
                weights['content'] * content_norm.get(item_id, 0) +
                weights['popularity'] * pop_norm.get(item_id, 0)
            )
        
        sorted_items = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return sorted_items
    
    def explain(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """Explain hybrid recommendation with component contributions."""
        self._validate_fitted()
        
        is_cold = self._is_cold_start_user(user_id)
        
        # Get individual model explanations
        contributions = []
        
        if not is_cold:
            svd_exp = self._svd.explain(user_id, item_id)
            contributions.append({
                "model": "Matrix Factorization",
                "weight": self.collaborative_weight,
                "reason": svd_exp.get("reason", "Latent factor match")
            })
            
            itemcf_exp = self._item_cf.explain(user_id, item_id)
            if itemcf_exp.get("source_items"):
                contributions.append({
                    "model": "Item Similarity",
                    "weight": self.collaborative_weight,
                    "reason": itemcf_exp.get("reason", "Similar to items you liked")
                })
        
        # Content contribution
        user_prefs = self._user_category_prefs.get(user_id, {})
        item_cat = None
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_cat = item_row.iloc[0]['category']
        
        if item_cat and item_cat in user_prefs:
            contributions.append({
                "model": "Content Match",
                "weight": self.content_weight if not is_cold else 0.6,
                "reason": f"Matches your preference for {item_cat}"
            })
        
        # Popularity contribution
        pop_exp = self._popularity.explain(user_id, item_id)
        contributions.append({
            "model": "Trending",
            "weight": self.popularity_weight if not is_cold else 0.4,
            "reason": pop_exp.get("reason", "Popular item")
        })
        
        item_info = {}
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_info = item_row.iloc[0].to_dict()
        
        primary_reason = contributions[0]['reason'] if contributions else "Recommended for you"
        
        return {
            "model": self.name,
            "reason": primary_reason,
            "is_cold_start": is_cold,
            "components": contributions,
            "explanation_text": f"{primary_reason}. Combined {len(contributions)} signals.",
            "factors": [
                {"name": "Strategy", "value": "Cold-start" if is_cold else "Personalized"},
                {"name": "Signals Used", "value": str(len(contributions))},
                {"name": "Category", "value": item_info.get("category", "N/A")}
            ]
        }
