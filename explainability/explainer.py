# Explainability Module - Transparent Recommendations
# FAANG-style recommendation explanations

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ExplanationFactor:
    """A single factor contributing to a recommendation."""
    name: str
    value: Any
    contribution: float
    description: str


@dataclass
class RecommendationExplanation:
    """Complete explanation for a recommendation."""
    item_id: int
    item_title: str
    score: float
    rank: int
    primary_reason: str
    factors: List[ExplanationFactor]
    similar_items: List[Dict]
    similar_users: List[Dict]
    model_source: str
    confidence: float
    is_cold_start: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "item_id": self.item_id,
            "item_title": self.item_title,
            "score": self.score,
            "rank": self.rank,
            "primary_reason": self.primary_reason,
            "factors": [
                {
                    "name": f.name,
                    "value": str(f.value),
                    "contribution": f.contribution,
                    "description": f.description
                }
                for f in self.factors
            ],
            "similar_items": self.similar_items,
            "similar_users": self.similar_users,
            "model_source": self.model_source,
            "confidence": self.confidence,
            "is_cold_start": self.is_cold_start
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"Recommended: {self.item_title}\n"
            f"Reason: {self.primary_reason}\n"
            f"Confidence: {self.confidence:.0%}"
        )


class RecommendationExplainer:
    """Generate transparent explanations for recommendations.
    
    Creates FAANG-style explanations like:
    - "Because you watched X"
    - "Similar users enjoyed this"
    - "Trending in category Y"
    - "Matches your preference for Z"
    
    Features:
    - Multi-factor attribution
    - Confidence scoring
    - Template-based text generation
    """
    
    def __init__(self):
        self._items_df = None
        self._users_df = None
        self._user_history: Dict[int, List[int]] = {}
        self._item_popularity: Dict[int, float] = {}
        
    def set_data(
        self,
        items_df=None,
        users_df=None,
        interactions_df=None
    ) -> None:
        """Set data sources for explanations."""
        self._items_df = items_df
        self._users_df = users_df
        
        if interactions_df is not None:
            # Build user history
            for user_id in interactions_df['user_id'].unique():
                user_ints = interactions_df[interactions_df['user_id'] == user_id]
                self._user_history[user_id] = user_ints['item_id'].tolist()
            
            # Build popularity
            counts = interactions_df['item_id'].value_counts()
            total = counts.sum()
            self._item_popularity = (counts / total).to_dict()
    
    def explain_recommendation(
        self,
        model,
        user_id: int,
        item_id: int,
        rank: int = 0,
        score: float = 0.0
    ) -> RecommendationExplanation:
        """Generate comprehensive explanation for a recommendation.
        
        Args:
            model: The recommender model that made the prediction
            user_id: Target user
            item_id: Recommended item
            rank: Position in recommendation list
            score: Prediction score
            
        Returns:
            RecommendationExplanation with full details
        """
        # Get model explanation
        model_exp = model.explain(user_id, item_id)
        
        # Get item info
        item_title = f"Item #{item_id}"
        item_category = "Unknown"
        
        if self._items_df is not None:
            item_row = self._items_df[self._items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item_title = item_row.iloc[0].get('title', item_title)
                item_category = item_row.iloc[0].get('category', item_category)
        
        # Build factors
        factors = []
        
        for f in model_exp.get('factors', []):
            factors.append(ExplanationFactor(
                name=f.get('name', 'Factor'),
                value=f.get('value', 'N/A'),
                contribution=0.5,  # Default contribution
                description=f"Based on {f.get('name', 'analysis')}"
            ))
        
        # Add popularity factor
        popularity = self._item_popularity.get(item_id, 0)
        if popularity > 0.01:  # Top 1% popularity
            factors.append(ExplanationFactor(
                name="Trending",
                value=f"Top {popularity*100:.1f}%",
                contribution=0.3,
                description="Currently trending among users"
            ))
        
        # Similar items
        similar_items = []
        source_items = model_exp.get('source_items', [])
        for src in source_items[:3]:
            src_id = src.get('item_id')
            src_title = f"Item #{src_id}"
            
            if self._items_df is not None:
                src_row = self._items_df[self._items_df['item_id'] == src_id]
                if len(src_row) > 0:
                    src_title = src_row.iloc[0].get('title', src_title)
            
            similar_items.append({
                "item_id": src_id,
                "title": src_title,
                "similarity": src.get('similarity', 0)
            })
        
        # Similar users
        similar_users = model_exp.get('similar_users', [])[:3]
        
        # Determine primary reason
        primary_reason = model_exp.get('reason', 'Recommended for you')
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            score=score,
            model_name=model.name,
            n_factors=len(factors),
            is_cold_start=model_exp.get('is_cold_start', False)
        )
        
        return RecommendationExplanation(
            item_id=item_id,
            item_title=item_title,
            score=score,
            rank=rank,
            primary_reason=primary_reason,
            factors=factors,
            similar_items=similar_items,
            similar_users=similar_users,
            model_source=model.name,
            confidence=confidence,
            is_cold_start=model_exp.get('is_cold_start', False)
        )
    
    def _calculate_confidence(
        self,
        score: float,
        model_name: str,
        n_factors: int,
        is_cold_start: bool
    ) -> float:
        """Calculate confidence score for explanation."""
        # Base confidence from score (normalized)
        base_confidence = min(1.0, max(0.0, score / 5.0))
        
        # Boost for more factors
        factor_boost = min(0.2, n_factors * 0.05)
        
        # Penalty for cold-start
        cold_penalty = 0.3 if is_cold_start else 0.0
        
        # Model-specific adjustments
        model_adjustment = {
            "Hybrid": 0.1,
            "SVD": 0.05,
            "Item-CF": 0.05,
            "User-CF": 0.0,
            "Popularity": -0.1
        }.get(model_name, 0.0)
        
        confidence = base_confidence + factor_boost + model_adjustment - cold_penalty
        return max(0.1, min(0.99, confidence))
    
    def explain_batch(
        self,
        model,
        user_id: int,
        recommendations: List[Tuple[int, float]]
    ) -> List[RecommendationExplanation]:
        """Generate explanations for all recommendations."""
        explanations = []
        
        for rank, (item_id, score) in enumerate(recommendations, 1):
            exp = self.explain_recommendation(
                model=model,
                user_id=user_id,
                item_id=item_id,
                rank=rank,
                score=score
            )
            explanations.append(exp)
        
        return explanations
    
    def generate_narrative(self, explanation: RecommendationExplanation) -> str:
        """Generate narrative explanation text."""
        parts = [f"**{explanation.item_title}**"]
        parts.append(f"\n*{explanation.primary_reason}*")
        
        if explanation.similar_items:
            items = [s['title'] for s in explanation.similar_items[:2]]
            parts.append(f"\nSimilar to: {', '.join(items)}")
        
        if explanation.is_cold_start:
            parts.append("\n_We're still learning your preferences_")
        
        parts.append(f"\nConfidence: {explanation.confidence:.0%}")
        
        return "\n".join(parts)


def get_explanation_template(reason_type: str) -> str:
    """Get template for explanation type."""
    templates = {
        "similar_items": "Because you watched {items}",
        "similar_users": "Users like you enjoyed this",
        "category_match": "Matches your preference for {category}",
        "trending": "Trending in {category}",
        "personalized": "Tailored to your taste profile",
        "new_release": "New release you might like",
        "top_rated": "Highly rated by the community"
    }
    return templates.get(reason_type, "Recommended for you")
