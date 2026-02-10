# Item-Based Collaborative Filtering
# Scalable item similarity approach (Netflix-style)

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class ItemBasedCF(BaseRecommender):
    """Item-based Collaborative Filtering recommender.
    
    Computes item-item similarities and recommends items similar
    to what the user has already engaged with.
    
    Advantages over User-CF:
    - More stable (items don't change as fast as users)
    - More scalable (fewer items than users typically)
    - Better explainability ("Because you watched X")
    
    Algorithm:
    1. Build item-user interaction matrix
    2. Compute item-item similarity
    3. For each item user liked, find similar items
    4. Aggregate and rank by weighted similarity
    
    Complexity: O(I² × U) for similarity, O(n_interactions × K) per user
    """
    
    def __init__(
        self,
        k_neighbors: int = 30,
        similarity_metric: str = "cosine",
        min_support: int = 5
    ):
        super().__init__(
            name="Item-CF",
            description="Item-based collaborative filtering with similarity scoring"
        )
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.min_support = min_support
        
        # Learned structures
        self._item_user_matrix: Optional[np.ndarray] = None
        self._item_similarity: Optional[np.ndarray] = None
        self._user_item_matrix: Optional[sparse.csr_matrix] = None
        self._item_idx_map: Dict[int, int] = {}
        self._idx_item_map: Dict[int, int] = {}
        self._user_idx_map: Dict[int, int] = {}
        self._item_neighbors: Dict[int, List[Tuple[int, float]]] = {}
        self._items_df: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame = None,
        items: pd.DataFrame = None
    ) -> "ItemBasedCF":
        """Build item-item similarity matrix."""
        
        print(f"[TRAIN] Training {self.name}...")
        
        self._items_df = items
        
        # Create mappings
        unique_users = sorted(interactions['user_id'].unique())
        unique_items = sorted(interactions['item_id'].unique())
        
        self._user_idx_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self._item_idx_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self._idx_item_map = {idx: iid for iid, idx in self._item_idx_map.items()}
        
        self._n_users = len(unique_users)
        self._n_items = len(unique_items)
        
        # Build user-item matrix
        type_weights = {'view': 1.0, 'click': 2.0, 'like': 4.0, 'purchase': 5.0}
        
        rows, cols, values = [], [], []
        for _, row in interactions.iterrows():
            user_idx = self._user_idx_map.get(row['user_id'])
            item_idx = self._item_idx_map.get(row['item_id'])
            
            if user_idx is not None and item_idx is not None:
                weight = type_weights.get(row['interaction_type'], 1.0)
                rows.append(user_idx)
                cols.append(item_idx)
                values.append(weight)
        
        self._user_item_matrix = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(self._n_users, self._n_items)
        ).tocsr()
        
        # Build item-user matrix (transposed)
        self._item_user_matrix = self._user_item_matrix.T.toarray()
        
        # Compute item similarity
        print("   Computing item similarities...")
        
        self._item_similarity = cosine_similarity(self._item_user_matrix)
        np.fill_diagonal(self._item_similarity, 0)
        
        # Pre-compute item neighbors
        self._compute_item_neighbors()
        
        self._is_fitted = True
        print(f"   [OK] {self.name} trained on {self._n_items} items")
        
        return self
    
    def _compute_item_neighbors(self) -> None:
        """Pre-compute K most similar items for each item."""
        for item_idx in range(self._n_items):
            similarities = self._item_similarity[item_idx]
            
            # Get top K similar items
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
            
            neighbors = [
                (self._idx_item_map[idx], similarities[idx])
                for idx in top_k_indices
                if similarities[idx] > 0
            ]
            
            item_id = self._idx_item_map[item_idx]
            self._item_neighbors[item_id] = neighbors
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Recommend items similar to user's history."""
        self._validate_fitted()
        
        user_idx = self._user_idx_map.get(user_id)
        
        if user_idx is None:
            return []
        
        # Get user's interacted items
        user_vector = self._user_item_matrix[user_idx].toarray().flatten()
        interacted_items = np.where(user_vector > 0)[0]
        
        if len(interacted_items) == 0:
            return []
        
        # Aggregate similar items
        item_scores = defaultdict(float)
        
        for item_idx in interacted_items:
            user_rating = user_vector[item_idx]
            item_id = self._idx_item_map[item_idx]
            
            for similar_item_id, similarity in self._item_neighbors.get(item_id, []):
                similar_item_idx = self._item_idx_map.get(similar_item_id)
                
                if similar_item_idx is not None:
                    # Weight by user's rating of original item
                    item_scores[similar_item_id] += similarity * user_rating
        
        # Exclude seen items
        if exclude_seen:
            seen_items = {self._idx_item_map[idx] for idx in interacted_items}
            item_scores = {k: v for k, v in item_scores.items() if k not in seen_items}
        
        # Sort and return top-N
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return sorted_items
    
    def get_similar_items(
        self,
        item_id: int,
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """Get most similar items."""
        self._validate_fitted()
        neighbors = self._item_neighbors.get(item_id, [])
        return neighbors[:n]
    
    def explain(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """Explain recommendation based on item similarity."""
        self._validate_fitted()
        
        user_idx = self._user_idx_map.get(user_id)
        target_item_idx = self._item_idx_map.get(item_id)
        
        source_items = []
        
        if user_idx is not None:
            user_vector = self._user_item_matrix[user_idx].toarray().flatten()
            interacted_items = np.where(user_vector > 0)[0]
            
            # Find which items triggered this recommendation
            for src_idx in interacted_items:
                src_item_id = self._idx_item_map[src_idx]
                neighbors = self._item_neighbors.get(src_item_id, [])
                
                for neighbor_id, similarity in neighbors:
                    if neighbor_id == item_id:
                        source_items.append({
                            "item_id": src_item_id,
                            "similarity": round(similarity, 3),
                            "your_engagement": user_vector[src_idx]
                        })
        
        # Sort by similarity
        source_items = sorted(source_items, key=lambda x: x['similarity'], reverse=True)[:3]
        
        # Get item titles
        if self._items_df is not None:
            for src in source_items:
                item_row = self._items_df[self._items_df['item_id'] == src['item_id']]
                if len(item_row) > 0:
                    src['title'] = item_row.iloc[0]['title']
        
        item_info = {}
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_info = item_row.iloc[0].to_dict()
        
        because_text = ""
        if source_items:
            src_title = source_items[0].get('title', f"Item #{source_items[0]['item_id']}")
            because_text = f"Because you watched: {src_title}"
        
        return {
            "model": self.name,
            "reason": because_text,
            "source_items": source_items,
            "explanation_text": because_text if because_text else "Similar to items you've engaged with",
            "factors": [
                {"name": "Based on", "value": f"{len(source_items)} items you liked"},
                {"name": "Top Match", "value": f"{source_items[0]['similarity']:.1%}" if source_items else "N/A"},
                {"name": "Category", "value": item_info.get("category", "N/A")}
            ]
        }
