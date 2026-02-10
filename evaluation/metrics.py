# Evaluation Metrics - Comprehensive Recommendation Evaluation
# Industry-standard metrics for ranking and recommendation quality

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricResult:
    """Container for a single metric result."""
    name: str
    value: float
    at_k: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        k_str = f"@{self.at_k}" if self.at_k else ""
        return f"{self.name}{k_str}: {self.value:.4f}"


class RecommenderMetrics:
    """Comprehensive metrics for recommendation system evaluation.
    
    Implements:
    - Precision@K: Fraction of recommendations that are relevant
    - Recall@K: Fraction of relevant items that are recommended
    - F1@K: Harmonic mean of precision and recall
    - MAP: Mean Average Precision
    - NDCG@K: Normalized Discounted Cumulative Gain
    - Hit Rate@K: Fraction of users with at least one hit
    - Coverage: Catalog coverage of recommendations
    - Diversity: Average pairwise distance between recommendations
    - Novelty: Inverse popularity-weighted recommendations
    """
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [5, 10, 20]
    
    def precision_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Precision@K: What fraction of top-K recommendations are relevant?"""
        if k <= 0:
            return 0.0
        
        top_k = recommended[:k]
        hits = sum(1 for r in top_k if r in relevant)
        return hits / k
    
    def recall_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Recall@K: What fraction of relevant items are in top-K?"""
        if len(relevant) == 0:
            return 0.0
        
        top_k = set(recommended[:k])
        hits = len(top_k & relevant)
        return hits / len(relevant)
    
    def f1_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """F1@K: Harmonic mean of Precision@K and Recall@K."""
        p = self.precision_at_k(recommended, relevant, k)
        r = self.recall_at_k(recommended, relevant, k)
        
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def average_precision(
        self,
        recommended: List[int],
        relevant: Set[int]
    ) -> float:
        """Average Precision for a single user."""
        if len(relevant) == 0:
            return 0.0
        
        ap = 0.0
        hits = 0
        
        for i, item in enumerate(recommended):
            if item in relevant:
                hits += 1
                ap += hits / (i + 1)
        
        return ap / len(relevant)
    
    def ndcg_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int,
        relevance_scores: Dict[int, float] = None
    ) -> float:
        """Normalized Discounted Cumulative Gain@K."""
        if k <= 0 or len(relevant) == 0:
            return 0.0
        
        # DCG
        dcg = 0.0
        for i, item in enumerate(recommended[:k]):
            if item in relevant:
                rel = relevance_scores.get(item, 1.0) if relevance_scores else 1.0
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Ideal DCG
        ideal_rels = sorted(
            [relevance_scores.get(r, 1.0) if relevance_scores else 1.0 for r in relevant],
            reverse=True
        )[:k]
        
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_rels)
        )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """Hit Rate: 1 if any relevant item in top-K, 0 otherwise."""
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    def coverage(
        self,
        all_recommendations: List[List[int]],
        catalog_size: int
    ) -> float:
        """Catalog Coverage: Fraction of items ever recommended."""
        if catalog_size == 0:
            return 0.0
        
        unique_items = set()
        for recs in all_recommendations:
            unique_items.update(recs)
        
        return len(unique_items) / catalog_size
    
    def diversity(
        self,
        recommended: List[int],
        item_vectors: Dict[int, np.ndarray]
    ) -> float:
        """Intra-List Diversity: Average pairwise distance."""
        if len(recommended) < 2:
            return 0.0
        
        vectors = []
        for item in recommended:
            if item in item_vectors:
                vectors.append(item_vectors[item])
        
        if len(vectors) < 2:
            return 0.0
        
        # Compute pairwise cosine distances
        total_distance = 0.0
        n_pairs = 0
        
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                # Cosine distance = 1 - cosine similarity
                similarity = np.dot(vectors[i], vectors[j])
                norm_i = np.linalg.norm(vectors[i])
                norm_j = np.linalg.norm(vectors[j])
                
                if norm_i > 0 and norm_j > 0:
                    cos_sim = similarity / (norm_i * norm_j)
                    total_distance += (1 - cos_sim)
                    n_pairs += 1
        
        return total_distance / n_pairs if n_pairs > 0 else 0.0
    
    def novelty(
        self,
        recommended: List[int],
        item_popularity: Dict[int, float]
    ) -> float:
        """Novelty: Average inverse popularity of recommendations."""
        if len(recommended) == 0:
            return 0.0
        
        total_novelty = 0.0
        
        for item in recommended:
            pop = item_popularity.get(item, 1e-10)
            # Self-information: -log(popularity)
            total_novelty += -np.log2(pop + 1e-10)
        
        return total_novelty / len(recommended)
    
    def evaluate_user(
        self,
        recommended: List[int],
        relevant: Set[int],
        k_values: List[int] = None
    ) -> Dict[str, float]:
        """Evaluate recommendations for a single user."""
        k_values = k_values or self.k_values
        metrics = {}
        
        for k in k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(recommended, relevant, k)
            metrics[f"recall@{k}"] = self.recall_at_k(recommended, relevant, k)
            metrics[f"f1@{k}"] = self.f1_at_k(recommended, relevant, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(recommended, relevant, k)
            metrics[f"hit_rate@{k}"] = self.hit_rate_at_k(recommended, relevant, k)
        
        metrics["map"] = self.average_precision(recommended, relevant)
        
        return metrics
    
    def evaluate_model(
        self,
        model,
        test_data: pd.DataFrame,
        n_recommendations: int = 10,
        item_vectors: Dict[int, np.ndarray] = None,
        item_popularity: Dict[int, float] = None,
        catalog_size: int = 0,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Evaluate a model on test data.
        
        Args:
            model: Fitted recommender model
            test_data: Test interactions DataFrame
            n_recommendations: Number of recommendations to generate
            item_vectors: Item feature vectors for diversity
            item_popularity: Item popularity scores for novelty
            catalog_size: Total items for coverage
            verbose: Print progress
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Build ground truth per user
        user_relevant = defaultdict(set)
        for _, row in test_data.iterrows():
            user_relevant[row['user_id']].add(row['item_id'])
        
        # Collect metrics per user
        all_metrics = defaultdict(list)
        all_recommendations = []
        
        users = list(user_relevant.keys())
        
        for i, user_id in enumerate(users):
            if verbose and (i + 1) % 500 == 0:
                print(f"   Evaluating user {i+1}/{len(users)}...")
            
            # Get recommendations
            try:
                recs = model.recommend(user_id, n=n_recommendations, exclude_seen=True)
                rec_items = [item_id for item_id, _ in recs]
            except Exception as e:
                continue
            
            if not rec_items:
                continue
            
            all_recommendations.append(rec_items)
            
            # Evaluate
            relevant = user_relevant[user_id]
            user_metrics = self.evaluate_user(rec_items, relevant)
            
            for metric_name, value in user_metrics.items():
                all_metrics[metric_name].append(value)
            
            # Diversity and novelty
            if item_vectors:
                div = self.diversity(rec_items, item_vectors)
                all_metrics["diversity"].append(div)
            
            if item_popularity:
                nov = self.novelty(rec_items, item_popularity)
                all_metrics["novelty"].append(nov)
        
        # Aggregate
        results = {}
        for metric_name, values in all_metrics.items():
            results[metric_name] = np.mean(values) if values else 0.0
        
        # Add coverage
        if catalog_size > 0 and all_recommendations:
            results["coverage"] = self.coverage(all_recommendations, catalog_size)
        
        return results


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a readable table."""
    lines = ["=" * 50, "EVALUATION METRICS", "=" * 50]
    
    # Group metrics
    ranking_metrics = {}
    other_metrics = {}
    
    for name, value in sorted(metrics.items()):
        if any(x in name for x in ['precision', 'recall', 'f1', 'ndcg', 'hit_rate', 'map']):
            ranking_metrics[name] = value
        else:
            other_metrics[name] = value
    
    lines.append("\nRanking Metrics:")
    lines.append("-" * 40)
    for name, value in sorted(ranking_metrics.items()):
        lines.append(f"  {name:<25} {value:.4f}")
    
    if other_metrics:
        lines.append("\nBeyond-Accuracy Metrics:")
        lines.append("-" * 40)
        for name, value in sorted(other_metrics.items()):
            lines.append(f"  {name:<25} {value:.4f}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)
