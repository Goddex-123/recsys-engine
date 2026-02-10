# RecSys Engine - Main Entry Point
"""
Production-grade recommendation system demonstrating FAANG-level
machine learning engineering patterns.

Usage:
    python main.py --mode dashboard    # Launch Streamlit dashboard
    python main.py --mode train        # Train models only
    python main.py --mode evaluate     # Run full evaluation
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = project_root / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def run_training():
    """Train all models and save to disk."""
    print("="*60)
    print("🚀 RecSys Engine - Training Pipeline")
    print("="*60)
    
    from data import DataLoader
    from models import (
        PopularityRecommender, UserBasedCF, ItemBasedCF,
        SVDRecommender, HybridRecommender
    )
    
    # Load or generate data
    print("\n📦 Loading data...")
    loader = DataLoader(data_dir=str(project_root / "generated_data"))
    users_df, items_df, interactions_df = loader.load_or_generate(
        n_users=5000,
        n_items=2000,
        n_interactions=100000
    )
    
    print(f"   Users: {len(users_df):,}")
    print(f"   Items: {len(items_df):,}")
    print(f"   Interactions: {len(interactions_df):,}")
    
    # Train models
    models = [
        ("Popularity", PopularityRecommender()),
        ("User-CF", UserBasedCF(k_neighbors=30)),
        ("Item-CF", ItemBasedCF(k_neighbors=20)),
        ("SVD", SVDRecommender(n_factors=30)),
        ("Hybrid", HybridRecommender())
    ]
    
    print("\n🤖 Training models...")
    for name, model in models:
        print(f"   Training {name}...", end=" ")
        model.fit(interactions_df, users_df, items_df)
        print("✓")
    
    print("\n✅ Training complete!")
    return loader, models


def run_evaluation():
    """Run full model evaluation."""
    print("="*60)
    print("📊 RecSys Engine - Evaluation Pipeline")
    print("="*60)
    
    # Train models first
    loader, model_list = run_training()
    
    from evaluation import ModelComparator
    
    # Split data
    print("\n📈 Evaluating models...")
    train_df, test_df = loader.train_test_split(test_ratio=0.2)
    
    # Get dataframes
    users_df, items_df, _ = loader.load_or_generate()
    
    # Compare
    comparator = ModelComparator()
    comparison = comparator.compare_models(
        models=[m for _, m in model_list],
        train_data=train_df,
        test_data=test_df,
        users_df=users_df,
        items_df=items_df,
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("📋 Evaluation Summary")
    print("="*60)
    
    for model_name in comparison.model_names:
        metrics = comparison.metrics.get(model_name, {})
        print(f"\n{model_name}:")
        print(f"   Precision@10: {metrics.get('precision@10', 0):.1%}")
        print(f"   Recall@10:    {metrics.get('recall@10', 0):.1%}")
        print(f"   NDCG@10:      {metrics.get('ndcg@10', 0):.1%}")
        print(f"   MAP:          {metrics.get('map', 0):.1%}")
        print(f"   Coverage:     {metrics.get('coverage', 0):.1%}")
    
    print("\n✅ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="RecSys Engine - Production Recommendation System"
    )
    parser.add_argument(
        "--mode",
        choices=["dashboard", "train", "evaluate"],
        default="dashboard",
        help="Operation mode"
    )
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "train":
        run_training()
    elif args.mode == "evaluate":
        run_evaluation()


if __name__ == "__main__":
    main()
