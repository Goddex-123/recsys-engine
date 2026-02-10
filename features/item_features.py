# Item Feature Extraction
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime


class ItemFeatureExtractor:
    """Extract features for items.
    
    Features:
    - Popularity metrics
    - Freshness score
    - Category embeddings
    """
    
    def __init__(self, freshness_halflife_days: int = 90):
        self._item_features: Dict[int, Dict] = {}
        self._freshness_halflife = freshness_halflife_days
        self._category_list: List[str] = []
        self._tag_list: List[str] = []
    
    def fit(
        self,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame = None
    ) -> "ItemFeatureExtractor":
        """Extract features from item and interaction data."""
        
        now = datetime.now()
        
        # Build category and tag lists
        self._category_list = sorted(items_df['category'].unique().tolist())
        
        # Extract all tags
        all_tags = set()
        for tags in items_df['tags']:
            if isinstance(tags, list):
                all_tags.update(tags)
            elif isinstance(tags, str):
                try:
                    import ast
                    tag_list = ast.literal_eval(tags)
                    all_tags.update(tag_list)
                except:
                    pass
        self._tag_list = sorted(list(all_tags))
        
        # Interaction counts if available
        interaction_counts = {}
        if interactions_df is not None:
            interaction_counts = interactions_df['item_id'].value_counts().to_dict()
        
        # Build category index
        cat_to_idx = {cat: i for i, cat in enumerate(self._category_list)}
        tag_to_idx = {tag: i for i, tag in enumerate(self._tag_list)}
        
        for _, row in items_df.iterrows():
            item_id = row['item_id']
            
            # Popularity
            n_interactions = interaction_counts.get(item_id, 0)
            
            # Freshness
            try:
                if pd.notna(row.get('release_date')):
                    release = pd.to_datetime(row['release_date'])
                    days_old = (now - release).days
                    freshness = 2 ** (-days_old / self._freshness_halflife)
                else:
                    freshness = 0.5
            except:
                freshness = 0.5
            
            # Rating
            avg_rating = row.get('avg_rating', 3.0)
            rating_count = row.get('rating_count', 0)
            
            # Category one-hot
            category_vector = np.zeros(len(self._category_list))
            if row['category'] in cat_to_idx:
                category_vector[cat_to_idx[row['category']]] = 1.0
            
            # Tag multi-hot
            tag_vector = np.zeros(len(self._tag_list))
            tags = row.get('tags', [])
            if isinstance(tags, str):
                try:
                    import ast
                    tags = ast.literal_eval(tags)
                except:
                    tags = []
            if isinstance(tags, list):
                for tag in tags:
                    if tag in tag_to_idx:
                        tag_vector[tag_to_idx[tag]] = 1.0
            
            # Combined content vector
            content_vector = np.concatenate([category_vector, tag_vector])
            norm = np.linalg.norm(content_vector)
            if norm > 0:
                content_vector = content_vector / norm
            
            self._item_features[item_id] = {
                'n_interactions': n_interactions,
                'freshness': freshness,
                'avg_rating': avg_rating,
                'rating_count': rating_count,
                'category': row['category'],
                'category_vector': category_vector,
                'tag_vector': tag_vector,
                'content_vector': content_vector
            }
        
        return self
    
    def get_features(self, item_id: int) -> Optional[Dict]:
        """Get extracted features for an item."""
        return self._item_features.get(item_id)
    
    def get_popularity_score(self, item_id: int) -> float:
        """Get normalized popularity score."""
        features = self._item_features.get(item_id)
        if not features:
            return 0.0
        
        max_interactions = max(
            f['n_interactions'] for f in self._item_features.values()
        ) or 1
        
        return features['n_interactions'] / max_interactions
    
    def get_content_vector(self, item_id: int) -> Optional[np.ndarray]:
        """Get content feature vector for item."""
        features = self._item_features.get(item_id)
        if features:
            return features['content_vector']
        return None
    
    def content_similarity(self, item_id1: int, item_id2: int) -> float:
        """Compute content similarity between items."""
        vec1 = self.get_content_vector(item_id1)
        vec2 = self.get_content_vector(item_id2)
        
        if vec1 is None or vec2 is None:
            return 0.0
        
        return float(np.dot(vec1, vec2))
