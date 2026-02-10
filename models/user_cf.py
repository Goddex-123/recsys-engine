# User-Based Collaborative Filtering
# Classic kNN approach with similarity-based recommendations

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender


class UserBasedCF(BaseRecommender):
    """User-based Collaborative Filtering recommender.
    
    Finds similar users based on interaction patterns and recommends
    items those similar users have engaged with.
    
    Algorithm:
    1. Build user-item interaction matrix
    2. Compute user-user similarity (cosine/pearson)
    3. For target user, find K nearest neighbors
    4. Aggregate neighbor preferences for recommendations
    
    Complexity: O(U² × I) for similarity, O(K × I) per recommendation
    """
    
    def __init__(
        self,
        k_neighbors: int = 50,
        min_common_items: int = 3,
        similarity_metric: str = "cosine"
    ):
        super().__init__(
            name="User-CF",
            description="User-based collaborative filtering with kNN"
        )
        self.k_neighbors = k_neighbors
        self.min_common_items = min_common_items
        self.similarity_metric = similarity_metric
        
        # Learned structures
        self._user_item_matrix: Optional[sparse.csr_matrix] = None
        self._user_similarity: Optional[np.ndarray] = None
        self._user_idx_map: Dict[int, int] = {}
        self._idx_user_map: Dict[int, int] = {}
        self._item_idx_map: Dict[int, int] = {}
        self._idx_item_map: Dict[int, int] = {}
        self._user_neighbors: Dict[int, List[Tuple[int, float]]] = {}
        self._items_df: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame = None,
        items: pd.DataFrame = None
    ) -> "UserBasedCF":
        """Build user-item matrix and compute similarities."""
        
        print(f"[TRAIN] Training {self.name}...")
        
        self._items_df = items
        
        # Create ID mappings
        unique_users = sorted(interactions['user_id'].unique())
        unique_items = sorted(interactions['item_id'].unique())
        
        self._user_idx_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self._idx_user_map = {idx: uid for uid, idx in self._user_idx_map.items()}
        self._item_idx_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self._idx_item_map = {idx: iid for iid, idx in self._item_idx_map.items()}
        
        self._n_users = len(unique_users)
        self._n_items = len(unique_items)
        
        # Build interaction matrix with implicit weights
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
        
        # Aggregate duplicates
        self._user_item_matrix = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(self._n_users, self._n_items)
        ).tocsr()
        
        # Compute user similarity
        print("   Computing user similarities...")
        
        # Normalize for cosine similarity
        matrix_dense = self._user_item_matrix.toarray()
        
        # Only compute if not too large
        if self._n_users <= 20000:
            self._user_similarity = cosine_similarity(matrix_dense)
            # Zero out diagonal
            np.fill_diagonal(self._user_similarity, 0)
        else:
            # For large datasets, compute on-demand
            self._user_similarity = None
        
        # Pre-compute nearest neighbors
        self._compute_neighbors()
        
        self._is_fitted = True
        print(f"   [OK] {self.name} trained on {self._n_users} users, {self._n_items} items")
        
        return self
    
    def _compute_neighbors(self) -> None:
        """Pre-compute K nearest neighbors for each user."""
        if self._user_similarity is None:
            return
        
        for user_idx in range(self._n_users):
            similarities = self._user_similarity[user_idx]
            
            # Get top K (excluding user itself)
            top_k_indices = np.argsort(similarities)[-self.k_neighbors:][::-1]
            
            neighbors = [
                (self._idx_user_map[idx], similarities[idx])
                for idx in top_k_indices
                if similarities[idx] > 0
            ]
            
            user_id = self._idx_user_map[user_idx]
            self._user_neighbors[user_id] = neighbors
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate recommendations based on similar users."""
        self._validate_fitted()
        
        user_idx = self._user_idx_map.get(user_id)
        
        # Cold-start: return empty (caller should fallback to popularity)
        if user_idx is None:
            return []
        
        # Get neighbors
        neighbors = self._user_neighbors.get(user_id, [])
        
        if not neighbors:
            return []
        
        # Aggregate neighbor preferences
        user_vector = self._user_item_matrix[user_idx].toarray().flatten()
        seen_items = set(np.where(user_vector > 0)[0])
        
        item_scores = defaultdict(float)
        similarity_sums = defaultdict(float)
        
        for neighbor_id, similarity in neighbors:
            neighbor_idx = self._user_idx_map.get(neighbor_id)
            if neighbor_idx is None:
                continue
            
            neighbor_vector = self._user_item_matrix[neighbor_idx].toarray().flatten()
            
            for item_idx, rating in enumerate(neighbor_vector):
                if rating > 0:
                    item_scores[item_idx] += similarity * rating
                    similarity_sums[item_idx] += abs(similarity)
        
        # Normalize
        final_scores = {}
        for item_idx, score in item_scores.items():
            if exclude_seen and item_idx in seen_items:
                continue
            if similarity_sums[item_idx] > 0:
                final_scores[item_idx] = score / similarity_sums[item_idx]
        
        # Sort and map back to item IDs
        sorted_items = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [
            (self._idx_item_map[idx], score)
            for idx, score in sorted_items
        ]
    
    def get_similar_users(
        self,
        user_id: int,
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """Get most similar users."""
        self._validate_fitted()
        neighbors = self._user_neighbors.get(user_id, [])
        return neighbors[:n]
    
    def explain(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """Explain recommendation based on similar users."""
        self._validate_fitted()
        
        neighbors = self._user_neighbors.get(user_id, [])
        item_idx = self._item_idx_map.get(item_id)
        
        contributing_users = []
        
        for neighbor_id, similarity in neighbors[:5]:
            neighbor_idx = self._user_idx_map.get(neighbor_id)
            if neighbor_idx is not None and item_idx is not None:
                neighbor_vector = self._user_item_matrix[neighbor_idx].toarray().flatten()
                if neighbor_vector[item_idx] > 0:
                    contributing_users.append({
                        "user_id": neighbor_id,
                        "similarity": round(similarity, 3),
                        "interaction_strength": neighbor_vector[item_idx]
                    })
        
        item_info = {}
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_info = item_row.iloc[0].to_dict()
        
        return {
            "model": self.name,
            "reason": f"Users similar to you enjoyed this",
            "similar_users": contributing_users[:3],
            "total_neighbors": len(neighbors),
            "explanation_text": f"Based on {len(contributing_users)} similar users who enjoyed this item",
            "factors": [
                {"name": "Similar Users", "value": f"{len(contributing_users)} found"},
                {"name": "Avg Similarity", "value": f"{np.mean([u['similarity'] for u in contributing_users]):.1%}" if contributing_users else "N/A"},
                {"name": "Category", "value": item_info.get("category", "N/A")}
            ]
        }
