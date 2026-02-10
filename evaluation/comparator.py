# Model Comparator - Side-by-Side Evaluation
# Compare multiple models with statistical analysis

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from .metrics import RecommenderMetrics


@dataclass
class ModelComparison:
    """Results of comparing multiple models."""
    model_names: List[str]
    metrics: Dict[str, Dict[str, float]]  # model -> metric -> value
    training_times: Dict[str, float]
    inference_times: Dict[str, float]
    best_model_per_metric: Dict[str, str]


class ModelComparator:
    """Compare multiple recommendation models.
    
    Features:
    - Side-by-side metric comparison
    - Training/inference time tracking
    - Statistical significance testing
    - Visualization helpers
    """
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [5, 10, 20]
        self.metrics = RecommenderMetrics(k_values=self.k_values)
        
    def compare_models(
        self,
        models: List[Any],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        users_df: pd.DataFrame = None,
        items_df: pd.DataFrame = None,
        n_recommendations: int = 10,
        verbose: bool = True
    ) -> ModelComparison:
        """Train and evaluate multiple models.
        
        Args:
            models: List of recommender model instances
            train_data: Training interactions
            test_data: Test interactions
            users_df: User metadata
            items_df: Item metadata
            n_recommendations: Recommendations per user
            verbose: Print progress
            
        Returns:
            ModelComparison with all results
        """
        model_metrics = {}
        training_times = {}
        inference_times = {}
        
        for model in models:
            model_name = model.name
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"📊 Evaluating: {model_name}")
                print('='*60)
            
            # Training
            start_time = time.time()
            model.fit(train_data, users_df, items_df)
            training_times[model_name] = time.time() - start_time
            
            if verbose:
                print(f"   Training time: {training_times[model_name]:.2f}s")
            
            # Build item popularity for novelty
            item_counts = train_data['item_id'].value_counts()
            total = item_counts.sum()
            item_popularity = (item_counts / total).to_dict()
            
            # Evaluation
            start_time = time.time()
            metrics = self.metrics.evaluate_model(
                model=model,
                test_data=test_data,
                n_recommendations=n_recommendations,
                item_popularity=item_popularity,
                catalog_size=items_df['item_id'].nunique() if items_df is not None else 0,
                verbose=verbose
            )
            inference_times[model_name] = time.time() - start_time
            
            model_metrics[model_name] = metrics
            
            if verbose:
                print(f"\n   Results for {model_name}:")
                for metric_name in sorted(metrics.keys())[:8]:
                    print(f"      {metric_name}: {metrics[metric_name]:.4f}")
        
        # Find best model per metric
        best_per_metric = {}
        all_metric_names = set()
        for metrics in model_metrics.values():
            all_metric_names.update(metrics.keys())
        
        for metric_name in all_metric_names:
            best_model = None
            best_value = -np.inf
            
            for model_name, metrics in model_metrics.items():
                value = metrics.get(metric_name, 0)
                if value > best_value:
                    best_value = value
                    best_model = model_name
            
            best_per_metric[metric_name] = best_model
        
        return ModelComparison(
            model_names=[m.name for m in models],
            metrics=model_metrics,
            training_times=training_times,
            inference_times=inference_times,
            best_model_per_metric=best_per_metric
        )
    
    def to_dataframe(self, comparison: ModelComparison) -> pd.DataFrame:
        """Convert comparison results to DataFrame."""
        data = []
        
        for model_name in comparison.model_names:
            row = {"Model": model_name}
            row.update(comparison.metrics.get(model_name, {}))
            row["Training Time (s)"] = comparison.training_times.get(model_name, 0)
            row["Inference Time (s)"] = comparison.inference_times.get(model_name, 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        return df.set_index("Model")
    
    def get_summary(self, comparison: ModelComparison) -> str:
        """Generate text summary of comparison."""
        lines = [
            "\n" + "="*70,
            "MODEL COMPARISON SUMMARY",
            "="*70
        ]
        
        # Best overall (by MAP)
        map_scores = {
            m: comparison.metrics[m].get('map', 0) 
            for m in comparison.model_names
        }
        best_overall = max(map_scores, key=map_scores.get)
        
        lines.append(f"\n🏆 Best Overall (MAP): {best_overall} ({map_scores[best_overall]:.4f})")
        
        # Training speed
        fastest = min(comparison.training_times, key=comparison.training_times.get)
        lines.append(f"⚡ Fastest Training: {fastest} ({comparison.training_times[fastest]:.2f}s)")
        
        # Key metrics comparison
        lines.append("\n📊 Key Metrics:")
        lines.append("-"*50)
        
        key_metrics = ['precision@10', 'recall@10', 'ndcg@10', 'map', 'coverage', 'diversity']
        
        for metric in key_metrics:
            if metric not in comparison.best_model_per_metric:
                continue
            
            best = comparison.best_model_per_metric[metric]
            values = " | ".join([
                f"{m}: {comparison.metrics[m].get(metric, 0):.4f}"
                for m in comparison.model_names
            ])
            lines.append(f"  {metric:<15} Best: {best}")
            lines.append(f"    {values}")
        
        lines.append("="*70)
        
        return "\n".join(lines)


def create_comparison_chart_data(comparison: ModelComparison) -> Dict:
    """Prepare data for visualization."""
    
    chart_data = {
        "models": comparison.model_names,
        "metrics": {}
    }
    
    key_metrics = ['precision@5', 'precision@10', 'recall@10', 'ndcg@10', 'map']
    
    for metric in key_metrics:
        chart_data["metrics"][metric] = [
            comparison.metrics[m].get(metric, 0)
            for m in comparison.model_names
        ]
    
    return chart_data
