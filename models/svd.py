# Matrix Factorization - SVD-based Recommender
# Latent factor model for implicit feedback (ALS-style)

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from scipy import sparse
from scipy.sparse.linalg import svds

from .base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """SVD-based Matrix Factorization recommender.
    
    Decomposes user-item matrix into latent factors:
    R ≈ U × Σ × V^T
    
    Where:
    - U: User latent factors (users × k)
    - Σ: Singular values (importance of each factor)
    - V: Item latent factors (items × k)
    
    Features:
    - Truncated SVD for dimensionality reduction
    - Handles implicit feedback
    - Efficient prediction via dot products
    
    Complexity: O(min(U,I) × k²) for training, O(k) per prediction
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        regularization: float = 0.01
    ):
        super().__init__(
            name="SVD",
            description="Matrix factorization using truncated SVD"
        )
        self.n_factors = n_factors
        self.regularization = regularization
        
        # Learned parameters
        self._user_factors: Optional[np.ndarray] = None
        self._item_factors: Optional[np.ndarray] = None
        self._singular_values: Optional[np.ndarray] = None
        self._user_idx_map: Dict[int, int] = {}
        self._idx_user_map: Dict[int, int] = {}
        self._item_idx_map: Dict[int, int] = {}
        self._idx_item_map: Dict[int, int] = {}
        self._user_item_matrix: Optional[sparse.csr_matrix] = None
        self._global_mean: float = 0.0
        self._user_bias: Optional[np.ndarray] = None
        self._item_bias: Optional[np.ndarray] = None
        self._items_df: Optional[pd.DataFrame] = None
        
    def fit(
        self,
        interactions: pd.DataFrame,
        users: pd.DataFrame = None,
        items: pd.DataFrame = None
    ) -> "SVDRecommender":
        """Train SVD model on interaction data."""
        
        print(f"[TRAIN] Training {self.name}...")
        
        self._items_df = items
        
        # Create mappings
        unique_users = sorted(interactions['user_id'].unique())
        unique_items = sorted(interactions['item_id'].unique())
        
        self._user_idx_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self._idx_user_map = {idx: uid for uid, idx in self._user_idx_map.items()}
        self._item_idx_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self._idx_item_map = {idx: iid for iid, idx in self._item_idx_map.items()}
        
        self._n_users = len(unique_users)
        self._n_items = len(unique_items)
        
        # Build interaction matrix
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
        
        # Compute biases
        matrix = self._user_item_matrix.toarray()
        non_zero_mask = matrix > 0
        
        self._global_mean = matrix[non_zero_mask].mean() if non_zero_mask.any() else 0
        
        # User and item biases
        self._user_bias = np.zeros(self._n_users)
        self._item_bias = np.zeros(self._n_items)
        
        for u in range(self._n_users):
            user_ratings = matrix[u, matrix[u] > 0]
            if len(user_ratings) > 0:
                self._user_bias[u] = user_ratings.mean() - self._global_mean
        
        for i in range(self._n_items):
            item_ratings = matrix[matrix[:, i] > 0, i]
            if len(item_ratings) > 0:
                self._item_bias[i] = item_ratings.mean() - self._global_mean
        
        # Center the matrix
        centered_matrix = matrix.copy()
        for u in range(self._n_users):
            for i in range(self._n_items):
                if centered_matrix[u, i] > 0:
                    centered_matrix[u, i] -= (
                        self._global_mean + 
                        self._user_bias[u] + 
                        self._item_bias[i]
                    )
        
        # Truncated SVD
        print("   Computing SVD decomposition...")
        
        k = min(self.n_factors, self._n_users - 1, self._n_items - 1)
        
        try:
            U, sigma, Vt = svds(
                sparse.csr_matrix(centered_matrix),
                k=k
            )
            
            # Sort by singular value importance
            idx = np.argsort(sigma)[::-1]
            self._singular_values = sigma[idx]
            self._user_factors = U[:, idx]
            self._item_factors = Vt[idx, :].T
            
        except Exception as e:
            print(f"   Warning: SVD failed ({e}), using random initialization")
            self._user_factors = np.random.randn(self._n_users, k) * 0.01
            self._item_factors = np.random.randn(self._n_items, k) * 0.01
            self._singular_values = np.ones(k)
        
        self._is_fitted = True
        print(f"   [OK] {self.name} trained with {k} latent factors")
        
        return self
    
    def predict_score(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair."""
        self._validate_fitted()
        
        user_idx = self._user_idx_map.get(user_id)
        item_idx = self._item_idx_map.get(item_id)
        
        if user_idx is None or item_idx is None:
            return self._global_mean
        
        # Prediction = global_mean + user_bias + item_bias + U·V^T
        latent_score = np.dot(
            self._user_factors[user_idx] * self._singular_values,
            self._item_factors[item_idx]
        )
        
        prediction = (
            self._global_mean +
            self._user_bias[user_idx] +
            self._item_bias[item_idx] +
            latent_score
        )
        
        return float(prediction)
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[int, float]]:
        """Generate top-N recommendations using latent factors."""
        self._validate_fitted()
        
        user_idx = self._user_idx_map.get(user_id)
        
        if user_idx is None:
            return []
        
        # Compute all predictions for this user
        user_vec = self._user_factors[user_idx] * self._singular_values
        all_scores = (
            self._global_mean +
            self._user_bias[user_idx] +
            self._item_bias +
            np.dot(self._item_factors, user_vec)
        )
        
        # Exclude seen items
        if exclude_seen:
            seen_mask = self._user_item_matrix[user_idx].toarray().flatten() > 0
            all_scores[seen_mask] = -np.inf
        
        # Get top-N
        top_indices = np.argsort(all_scores)[-n:][::-1]
        
        return [
            (self._idx_item_map[idx], float(all_scores[idx]))
            for idx in top_indices
            if all_scores[idx] > -np.inf
        ]
    
    def explain(
        self,
        user_id: int,
        item_id: int
    ) -> Dict[str, Any]:
        """Explain recommendation through latent factors."""
        self._validate_fitted()
        
        user_idx = self._user_idx_map.get(user_id)
        item_idx = self._item_idx_map.get(item_id)
        
        factors_contribution = []
        
        if user_idx is not None and item_idx is not None:
            user_vec = self._user_factors[user_idx]
            item_vec = self._item_factors[item_idx]
            
            # Top contributing factors
            contributions = user_vec * item_vec * self._singular_values
            top_factors = np.argsort(np.abs(contributions))[-5:][::-1]
            
            for f in top_factors:
                factors_contribution.append({
                    "factor": int(f),
                    "contribution": round(contributions[f], 4),
                    "user_affinity": round(user_vec[f], 4),
                    "item_loading": round(item_vec[f], 4)
                })
        
        predicted_score = self.predict_score(user_id, item_id)
        
        item_info = {}
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_info = item_row.iloc[0].to_dict()
        
        return {
            "model": self.name,
            "reason": "Matches your taste profile",
            "predicted_score": round(predicted_score, 3),
            "latent_factors": factors_contribution[:3],
            "n_factors": self.n_factors,
            "explanation_text": f"Based on {self.n_factors} latent taste dimensions, predicted fit: {predicted_score:.2f}",
            "factors": [
                {"name": "Predicted Fit", "value": f"{predicted_score:.2f}"},
                {"name": "Latent Dimensions", "value": str(self.n_factors)},
                {"name": "Category", "value": item_info.get("category", "N/A")}
            ]
        }
    
    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """Get user latent factor vector."""
        self._validate_fitted()
        user_idx = self._user_idx_map.get(user_id)
        if user_idx is not None:
            return self._user_factors[user_idx]
        return None
    
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """Get item latent factor vector."""
        self._validate_fitted()
        item_idx = self._item_idx_map.get(item_id)
        if item_idx is not None:
            return self._item_factors[item_idx]
        return None
