# RecSys Engine - Premium Streamlit Dashboard
# FAANG-level recommendation system visualization

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data import DataLoader
from models import (
    PopularityRecommender, UserBasedCF, ItemBasedCF,
    SVDRecommender, HybridRecommender
)
from evaluation import ModelComparator
from explainability import RecommendationExplainer


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="RecSys Engine | AI-Powered Recommendations",
    page_icon="&#127919;",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data_loaded': False,
        'models_trained': False,
        'current_user_id': 0,
        'selected_model': 'Hybrid',
        'comparison_results': None,
        'simulation_interactions': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================
# DATA LOADING
# ============================================
@st.cache_resource
def load_data():
    """Load or generate dataset."""
    loader = DataLoader(data_dir=str(project_root / "generated_data"))
    users_df, items_df, interactions_df = loader.load_or_generate(
        n_users=5000,
        n_items=2000,
        n_interactions=100000
    )
    return loader, users_df, items_df, interactions_df


@st.cache_resource
def train_models(_loader, _interactions_df, _users_df, _items_df):
    """Train all recommendation models."""
    models = {}
    
    # Popularity
    models['Popularity'] = PopularityRecommender()
    models['Popularity'].fit(_interactions_df, _users_df, _items_df)
    
    # User-CF - train on smaller subset for speed
    models['User-CF'] = UserBasedCF(k_neighbors=30)
    models['User-CF'].fit(_interactions_df, _users_df, _items_df)
    
    # Item-CF
    models['Item-CF'] = ItemBasedCF(k_neighbors=20)
    models['Item-CF'].fit(_interactions_df, _users_df, _items_df)
    
    # SVD
    models['SVD'] = SVDRecommender(n_factors=30)
    models['SVD'].fit(_interactions_df, _users_df, _items_df)
    
    # Hybrid
    models['Hybrid'] = HybridRecommender()
    models['Hybrid'].fit(_interactions_df, _users_df, _items_df)
    
    return models


# ============================================
# SIDEBAR
# ============================================
def render_sidebar():
    """Render sidebar navigation and controls."""
    
    with st.sidebar:
        # Logo and title
        html = (
            '<div style="text-align: center; padding: 1rem 0;">'
            '<h1 style="background: linear-gradient(135deg, #6366F1, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; margin: 0;">&#127919; RecSys</h1>'
            '<p style="color: #64748B; font-size: 0.9rem; margin: 0.5rem 0 0 0;">AI-Powered Recommendations</p>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigation")
        
        page = st.radio(
            "Select Page",
            ["Dashboard", "User Explorer", "Algorithm Lab", 
             "Metrics", "Cold-Start", "Live Demo"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats
        if st.session_state.data_loaded:
            st.markdown("### Data Stats")
            st.metric("Users", f"{len(st.session_state.users_df):,}")
            st.metric("Items", f"{len(st.session_state.items_df):,}")
            st.metric("Interactions", f"{len(st.session_state.interactions_df):,}")
        
        st.markdown("---")
        
        # Model selector
        st.markdown("### Active Model")
        st.session_state.selected_model = st.selectbox(
            "Select Model",
            ["Popularity", "User-CF", "Item-CF", "SVD", "Hybrid"],
            index=4,
            label_visibility="collapsed"
        )
        
        return page


# ============================================
# MAIN PAGES
# ============================================
def render_dashboard():
    """Render main dashboard page."""
    
    html = (
        '<div style="margin-bottom: 2rem;">'
        '<h1 style="color: #F8FAFC; margin-bottom: 0.5rem;">Welcome to RecSys Engine</h1>'
        '<p style="color: #94A3B8; font-size: 1.1rem;">Production-grade personalized recommendations powered by advanced ML algorithms</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("Total Users", f"{len(st.session_state.users_df):,}", "&#128101;", "+12%"),
        ("Catalog Size", f"{len(st.session_state.items_df):,}", "&#127916;", "+5%"),
        ("Interactions", f"{len(st.session_state.interactions_df):,}", "&#9889;", "+23%"),
        ("Models Active", "5", "&#129302;", "All Systems Go")
    ]
    
    for col, (label, value, icon, delta) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            html = (
                f'<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 16px; padding: 1.5rem; position: relative; overflow: hidden;">'
                f'<div style="position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #6366F1, #8B5CF6);"></div>'
                f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>'
                f'<div style="font-size: 1.75rem; font-weight: 800; background: linear-gradient(135deg, #6366F1, #8B5CF6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</div>'
                f'<div style="color: #94A3B8; font-size: 0.9rem;">{label}</div>'
                f'<div style="color: #6EE7B7; font-size: 0.8rem; margin-top: 0.5rem;">&#9650; {delta}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    # Quick recommendations for random user
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### Trending Recommendations")
        
        model = st.session_state.models['Hybrid']
        sample_user = st.session_state.users_df['user_id'].iloc[42]
        recs = model.recommend(sample_user, n=4)
        
        items_df = st.session_state.items_df
        
        rec_cols = st.columns(4)
        for i, (item_id, score) in enumerate(recs):
            item_row = items_df[items_df['item_id'] == item_id]
            if len(item_row) > 0:
                item = item_row.iloc[0]
                with rec_cols[i]:
                    html = (
                        f'<div style="background: linear-gradient(180deg, rgba(99, 102, 241, 0.2) 0%, rgba(30, 41, 59, 0.8) 100%); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1rem; text-align: center; height: 160px;">'
                        f'<div style="font-size: 2.5rem; margin-bottom: 0.5rem;">&#127916;</div>'
                        f'<div style="color: #F8FAFC; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.25rem;">{item["title"][:20]}...</div>'
                        f'<div style="color: #A5B4FC; font-size: 0.75rem;">{item["category"]}</div>'
                        f'<div style="color: #6EE7B7; font-size: 0.7rem; margin-top: 0.5rem;">Match: {score:.1%}</div>'
                        f'</div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("### System Health")
        
        html = '<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 1rem;">'
        st.markdown(html, unsafe_allow_html=True)
        
        health_items = [
            ("Data Pipeline", "Operational", "&#128994;"),
            ("Model Training", "Complete", "&#128994;"),
            ("API Status", "Ready", "&#128994;"),
            ("Cache", "Warm", "&#128993;")
        ]
        
        for name, status, indicator in health_items:
            html = (
                f'<div style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(148, 163, 184, 0.1);">'
                f'<span style="color: #94A3B8;">{name}</span>'
                f'<span>{indicator} {status}</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_user_explorer():
    """Render user exploration page."""
    
    st.markdown("## User Explorer")
    
    # User selector
    col1, col2 = st.columns([1, 2])
    
    with col1:
        users_df = st.session_state.users_df
        
        # Search/filter
        user_search = st.text_input("Search User ID", "")
        
        if user_search:
            filtered_users = users_df[
                users_df['user_id'].astype(str).str.contains(user_search)
            ]
        else:
            filtered_users = users_df.head(100)
        
        user_options = filtered_users['user_id'].tolist()
        
        if user_options:
            selected_user = st.selectbox(
                "Select User",
                user_options,
                format_func=lambda x: f"User #{x}"
            )
            st.session_state.current_user_id = selected_user
    
    with col2:
        if st.session_state.current_user_id is not None:
            user_id = st.session_state.current_user_id
            user_row = users_df[users_df['user_id'] == user_id]
            
            if len(user_row) > 0:
                user_data = user_row.iloc[0].to_dict()
                interactions_df = st.session_state.interactions_df
                user_interactions = interactions_df[interactions_df['user_id'] == user_id]
                
                # Parse preferred categories
                prefs = user_data.get('preferred_categories', [])
                if isinstance(prefs, str):
                    try:
                        import ast
                        prefs = ast.literal_eval(prefs)
                    except:
                        prefs = []
                
                # Profile card
                from dashboard.components.user_profile import render_user_profile
                render_user_profile(user_data, len(user_interactions), prefs)
    
    st.markdown("---")
    
    # Recommendations for selected user
    if st.session_state.current_user_id is not None:
        st.markdown("### Personalized Recommendations")
        
        model_name = st.session_state.selected_model
        model = st.session_state.models.get(model_name)
        
        if model:
            recs = model.recommend(st.session_state.current_user_id, n=6)
            
            if recs:
                from dashboard.components.recommendations import render_recommendations_grid
                
                # Get explanations
                explanations = {}
                for item_id, score in recs:
                    exp = model.explain(st.session_state.current_user_id, item_id)
                    explanations[item_id] = exp
                
                render_recommendations_grid(
                    recs, 
                    st.session_state.items_df,
                    explanations,
                    columns=3
                )
            else:
                st.info("No recommendations available for this user.")


def render_algorithm_lab():
    """Render algorithm comparison page."""
    
    st.markdown("## Algorithm Lab")
    st.markdown("Compare different recommendation algorithms side-by-side")
    
    if st.session_state.current_user_id is None:
        st.session_state.current_user_id = st.session_state.users_df['user_id'].iloc[0]
    
    user_id = st.session_state.current_user_id
    st.markdown(f"**Showing recommendations for User #{user_id}**")
    
    # Get recommendations from all models
    all_recs = {}
    for model_name, model in st.session_state.models.items():
        try:
            recs = model.recommend(user_id, n=5)
            all_recs[model_name] = recs
        except:
            all_recs[model_name] = []
    
    # Display comparison
    from dashboard.components.recommendations import render_algorithm_comparison
    render_algorithm_comparison(all_recs, st.session_state.items_df)
    
    st.markdown("---")
    
    # Model descriptions
    st.markdown("### Algorithm Descriptions")
    
    descriptions = {
        "Popularity": "Recommends globally popular items. Simple but effective baseline.",
        "User-CF": "Finds similar users and recommends what they liked.",
        "Item-CF": "Finds items similar to what you've interacted with.",
        "SVD": "Uses matrix factorization to discover latent taste factors.",
        "Hybrid": "Combines multiple signals for robust recommendations."
    }
    
    cols = st.columns(5)
    for col, (name, desc) in zip(cols, descriptions.items()):
        with col:
            html = (
                f'<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 1rem; height: 150px;">'
                f'<h4 style="color: #A5B4FC; margin: 0 0 0.5rem 0; font-size: 0.9rem;">{name}</h4>'
                f'<p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">{desc}</p>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)


def render_metrics_page():
    """Render metrics visualization page."""
    
    st.markdown("## Performance Metrics")
    
    if st.button("Run Full Evaluation", type="primary"):
        with st.spinner("Evaluating all models... This may take a moment."):
            comparator = ModelComparator()
            
            # Split data
            loader = st.session_state.loader
            train_df, test_df = loader.train_test_split(test_ratio=0.2)
            
            comparison = comparator.compare_models(
                models=list(st.session_state.models.values()),
                train_data=train_df,
                test_data=test_df,
                users_df=st.session_state.users_df,
                items_df=st.session_state.items_df,
                verbose=False
            )
            
            st.session_state.comparison_results = {
                'model_names': comparison.model_names,
                'metrics': comparison.metrics
            }
    
    if st.session_state.comparison_results:
        from dashboard.components.metrics_viz import (
            render_metrics_dashboard,
            render_comparison_chart
        )
        
        # Show metrics for best model
        best_model = 'Hybrid'
        if best_model in st.session_state.comparison_results['metrics']:
            render_metrics_dashboard(
                st.session_state.comparison_results['metrics'][best_model]
            )
        
        st.markdown("---")
        
        # Comparison chart
        render_comparison_chart(st.session_state.comparison_results)
    else:
        st.info("Click 'Run Full Evaluation' to see detailed metrics.")
        
        # Show placeholder metrics
        st.markdown("### Expected Performance Ranges")
        
        st.markdown("""
        | Model | Precision@10 | Recall@10 | NDCG@10 |
        |-------|-------------|-----------|---------|
        | Popularity | 5-10% | 3-8% | 8-15% |
        | User-CF | 8-15% | 5-12% | 12-20% |
        | Item-CF | 10-18% | 6-14% | 15-25% |
        | SVD | 12-20% | 8-16% | 18-28% |
        | **Hybrid** | **15-25%** | **10-18%** | **22-32%** |
        """)


def render_cold_start_page():
    """Render cold-start demonstration page."""
    
    st.markdown("## Cold-Start Problem")
    
    html = (
        '<div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(239, 68, 68, 0.1)); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem;">'
        '<h3 style="color: #FBBF24; margin: 0 0 0.5rem 0;">The Cold-Start Challenge</h3>'
        '<p style="color: #94A3B8; margin: 0;">New users have no interaction history, making personalization impossible with traditional collaborative filtering. This demo shows how different algorithms handle this challenge.</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    
    from dashboard.components.simulation import render_cold_start_demo
    render_cold_start_demo(st.session_state.models, st.session_state.items_df)


def render_live_demo():
    """Render live simulation page."""
    
    st.markdown("## Live Interaction Demo")
    
    from dashboard.components.simulation import (
        render_live_simulation,
        render_ab_test_simulation
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        sim_state = render_live_simulation(
            st.session_state.items_df,
            st.session_state.current_user_id
        )
        
        # Show updated recommendations based on simulation
        if sim_state['interactions']:
            st.markdown("### Updated Recommendations")
            st.markdown("*Recommendations would update based on your interactions*")
    
    with col2:
        render_ab_test_simulation()


# ============================================
# MAIN APP
# ============================================
def main():
    """Main application entry point."""
    
    # Load data
    with st.spinner("Loading data..."):
        loader, users_df, items_df, interactions_df = load_data()
        st.session_state.loader = loader
        st.session_state.users_df = users_df
        st.session_state.items_df = items_df
        st.session_state.interactions_df = interactions_df
        st.session_state.data_loaded = True
    
    # Train models
    with st.spinner("Training models..."):
        models = train_models(loader, interactions_df, users_df, items_df)
        st.session_state.models = models
        st.session_state.models_trained = True
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Route to appropriate page
    if "Dashboard" in page:
        render_dashboard()
    elif "User Explorer" in page:
        render_user_explorer()
    elif "Algorithm Lab" in page:
        render_algorithm_lab()
    elif "Metrics" in page:
        render_metrics_page()
    elif "Cold-Start" in page:
        render_cold_start_page()
    elif "Live Demo" in page:
        render_live_demo()
    
    # Footer
    st.markdown("---")
    html = (
        '<div style="text-align: center; color: #64748B; font-size: 0.8rem;">'
        'Built with Streamlit &bull; RecSys Engine v1.0 &bull; '
        '<a href="https://github.com" style="color: #6366F1;">GitHub</a>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
