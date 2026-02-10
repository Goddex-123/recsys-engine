# Data Loader - Efficient Data Loading and Caching
# Handles both generated and persisted data with caching

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import os
import pickle
from scipy import sparse

from .schemas import User, Item, Interaction, InteractionType
from .generator import DataGenerator


class DataLoader:
    """Efficient data loader with caching and preprocessing.
    
    Supports:
    - Loading from CSV files
    - In-memory data generation
    - Sparse matrix construction
    - Train/test splitting
    """
    
    def __init__(self, data_dir: str = "generated_data"):
        self.data_dir = data_dir
        self._users_df: Optional[pd.DataFrame] = None
        self._items_df: Optional[pd.DataFrame] = None
        self._interactions_df: Optional[pd.DataFrame] = None
        self._interaction_matrix: Optional[sparse.csr_matrix] = None
        
    def load_or_generate(
        self,
        n_users: int = 10_000,
        n_items: int = 5_000,
        n_interactions: int = 500_000,
        force_regenerate: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load existing data or generate new dataset."""
        
        users_path = os.path.join(self.data_dir, "users.csv")
        items_path = os.path.join(self.data_dir, "items.csv")
        interactions_path = os.path.join(self.data_dir, "interactions.csv")
        
        # Check if data exists
        if not force_regenerate and all(
            os.path.exists(p) for p in [users_path, items_path, interactions_path]
        ):
            print("[INFO] Loading existing data from CSV files...")
            self._users_df = pd.read_csv(users_path)
            self._items_df = pd.read_csv(items_path)
            self._interactions_df = pd.read_csv(interactions_path)
            print(f"   - Loaded {len(self._users_df):,} users, {len(self._items_df):,} items, {len(self._interactions_df):,} interactions")
        else:
            print("[INFO] Generating new dataset...")
            generator = DataGenerator(
                n_users=n_users,
                n_items=n_items,
                n_interactions=n_interactions
            )
            generator.generate_all()
            generator.save_to_csv(self.data_dir)
            self._users_df, self._items_df, self._interactions_df = generator.to_dataframes()
        
        return self._users_df, self._items_df, self._interactions_df
    
    @property
    def users(self) -> pd.DataFrame:
        """Get users DataFrame."""
        if self._users_df is None:
            self.load_or_generate()
        return self._users_df
    
    @property
    def items(self) -> pd.DataFrame:
        """Get items DataFrame."""
        if self._items_df is None:
            self.load_or_generate()
        return self._items_df
    
    @property
    def interactions(self) -> pd.DataFrame:
        """Get interactions DataFrame."""
        if self._interactions_df is None:
            self.load_or_generate()
        return self._interactions_df
    
    def build_interaction_matrix(
        self,
        value_type: str = "implicit"
    ) -> sparse.csr_matrix:
        """Build user-item interaction matrix.
        
        Args:
            value_type: 'implicit' (weighted scores), 'binary' (0/1), or 'rating'
        
        Returns:
            Sparse CSR matrix (users x items)
        """
        if self._interactions_df is None:
            self.load_or_generate()
        
        n_users = self._users_df['user_id'].max() + 1
        n_items = self._items_df['item_id'].max() + 1
        
        df = self._interactions_df.copy()
        
        # Calculate values based on type
        if value_type == "binary":
            values = np.ones(len(df))
        elif value_type == "rating":
            values = df['rating'].fillna(3.0).values
        else:  # implicit
            # Weight by interaction type
            type_weights = {
                'view': 1.0,
                'click': 2.0,
                'like': 4.0,
                'purchase': 5.0
            }
            values = df['interaction_type'].map(type_weights).values
        
        # Build sparse matrix
        rows = df['user_id'].values
        cols = df['item_id'].values
        
        # Aggregate duplicate interactions
        matrix = sparse.coo_matrix(
            (values, (rows, cols)),
            shape=(n_users, n_items)
        )
        
        # Convert to CSR for efficient row operations
        self._interaction_matrix = matrix.tocsr()
        
        return self._interaction_matrix
    
    def train_test_split(
        self,
        test_ratio: float = 0.2,
        by_time: bool = True,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split interactions into train and test sets.
        
        Args:
            test_ratio: Fraction of data for testing
            by_time: If True, split chronologically (realistic)
            random_seed: Random seed for reproducibility
        
        Returns:
            (train_df, test_df)
        """
        df = self.interactions.copy()
        
        if by_time:
            # Sort by timestamp and take recent interactions for test
            df = df.sort_values('timestamp')
            split_idx = int(len(df) * (1 - test_ratio))
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            # Random split
            np.random.seed(random_seed)
            mask = np.random.rand(len(df)) < (1 - test_ratio)
            train_df = df[mask]
            test_df = df[~mask]
        
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def get_user_history(self, user_id: int) -> pd.DataFrame:
        """Get all interactions for a specific user."""
        return self.interactions[self.interactions['user_id'] == user_id]
    
    def get_item_interactions(self, item_id: int) -> pd.DataFrame:
        """Get all interactions for a specific item."""
        return self.interactions[self.interactions['item_id'] == item_id]
    
    def get_user_dict(self, user_id: int) -> Dict:
        """Get user metadata as dictionary."""
        user_row = self.users[self.users['user_id'] == user_id]
        if len(user_row) == 0:
            return {}
        return user_row.iloc[0].to_dict()
    
    def get_item_dict(self, item_id: int) -> Dict:
        """Get item metadata as dictionary."""
        item_row = self.items[self.items['item_id'] == item_id]
        if len(item_row) == 0:
            return {}
        return item_row.iloc[0].to_dict()
    
    def get_statistics(self) -> Dict:
        """Calculate and return dataset statistics."""
        n_users = len(self.users)
        n_items = len(self.items)
        n_interactions = len(self.interactions)
        
        stats = {
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_interactions,
            "sparsity": 1 - (n_interactions / (n_users * n_items)),
            "avg_interactions_per_user": n_interactions / n_users,
            "avg_interactions_per_item": n_interactions / n_items,
            "interaction_types": self.interactions['interaction_type'].value_counts().to_dict(),
            "category_distribution": self.items['category'].value_counts().to_dict(),
        }
        
        return stats
