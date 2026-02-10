import streamlit as st
from typing import List, Dict, Tuple, Any


def render_recommendation_card(
    item: Dict,
    rank: int,
    score: float,
    explanation: Dict = None,
    show_explanation: bool = True
) -> None:
    """Render a single recommendation card with explanation."""
    
    title = item.get('title', f"Item #{item.get('item_id', 'N/A')}")
    category = item.get('category', 'Unknown')
    rating = item.get('avg_rating', 0)
    
    # Score percentage
    score_pct = min(100, max(0, score * 100 if score <= 1 else score * 10))
    
    # Rating stars
    stars = '&#11088;' * int(round(rating))
    
    # Explanation text
    reason = "Recommended for you"
    if explanation:
        reason = explanation.get('reason', explanation.get('primary_reason', reason))
    
    # Build HTML with ZERO indentation to prevent Markdown code block interpretation
    html = (
        f'<div style="background: rgba(30, 41, 59, 0.8); backdrop-filter: blur(20px); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 16px; padding: 1.25rem; margin-bottom: 1rem; transition: all 0.25s ease; position: relative; overflow: hidden;">'
        f'<div style="position: absolute; top: 0; left: 0; background: linear-gradient(135deg, #6366F1, #8B5CF6); color: white; padding: 0.25rem 0.75rem; border-radius: 0 0 12px 0; font-size: 0.8rem; font-weight: 700;">#{rank}</div>'
        f'<div style="display: flex; gap: 1rem; margin-top: 0.5rem;">'
        f'<div style="width: 80px; height: 100px; background: linear-gradient(135deg, #374151, #1F2937); border-radius: 8px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">'
        f'<span style="font-size: 2rem;">&#127916;</span>'
        f'</div>'
        f'<div style="flex: 1; min-width: 0;">'
        f'<h4 style="margin: 0 0 0.25rem 0; color: #F8FAFC; font-size: 1rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title}</h4>'
        f'<div style="display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.5rem;">'
        f'<span style="background: rgba(99, 102, 241, 0.2); color: #A5B4FC; padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.7rem; font-weight: 500;">{category}</span>'
        f'<span style="font-size: 0.75rem; color: #FCD34D;">{stars}</span>'
        f'</div>'
        f'<div style="margin-bottom: 0.5rem;">'
        f'<div style="height: 6px; background: rgba(51, 65, 85, 0.8); border-radius: 9999px; overflow: hidden;">'
        f'<div style="width: {score_pct}%; height: 100%; background: linear-gradient(90deg, #6366F1, #8B5CF6); border-radius: 9999px;"></div>'
        f'</div>'
        f'<div style="font-size: 0.7rem; color: #64748B; margin-top: 0.25rem;">Match Score: {score:.1%}</div>'
        f'</div>'
        f'<div style="font-size: 0.8rem; color: #94A3B8; font-style: italic;">&#128161; {reason}</div>'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    
    st.markdown(html, unsafe_allow_html=True)


def render_recommendations_grid(
    recommendations: List[Tuple[int, float]],
    items_df,
    explanations: Dict[int, Dict] = None,
    columns: int = 2
) -> None:
    """Render recommendations in a responsive grid."""
    
    if not recommendations:
        st.info("No recommendations available. Select a user with more activity.")
        return
    
    # Create item lookup
    item_lookup = {}
    if items_df is not None:
        for _, row in items_df.iterrows():
            item_lookup[row['item_id']] = row.to_dict()
    
    # Render in columns
    cols = st.columns(columns)
    
    for i, (item_id, score) in enumerate(recommendations):
        item = item_lookup.get(item_id, {'item_id': item_id})
        explanation = explanations.get(item_id, {}) if explanations else {}
        
        with cols[i % columns]:
            render_recommendation_card(
                item=item,
                rank=i + 1,
                score=score,
                explanation=explanation
            )


def render_algorithm_comparison(
    recommendations_by_model: Dict[str, List[Tuple[int, float]]],
    items_df
) -> None:
    """Render side-by-side algorithm comparison."""
    
    if not recommendations_by_model:
        st.warning("No recommendations to compare.")
        return
    
    model_names = list(recommendations_by_model.keys())
    
    # Model icons
    model_icons = {
        "Popularity": "&#128200;",
        "User-CF": "&#128101;",
        "Item-CF": "&#127916;",
        "SVD": "&#129518;",
        "Hybrid": "&#128256;"
    }
    
    # Tabs for each model
    tabs = st.tabs([f"{model_icons.get(m, '•')} {m}" for m in model_names])
    
    for tab, model_name in zip(tabs, model_names):
        with tab:
            recs = recommendations_by_model[model_name]
            
            if recs:
                render_recommendations_grid(recs[:6], items_df, columns=2)
            else:
                st.info(f"No recommendations from {model_name}")


def render_quick_picks(
    trending: List[Tuple[int, float]],
    items_df,
    max_items: int = 5
) -> None:
    """Render horizontal scrolling quick picks."""
    
    st.markdown("### Trending Now")
    
    cols = st.columns(max_items)
    
    item_lookup = {}
    if items_df is not None:
        for _, row in items_df.iterrows():
            item_lookup[row['item_id']] = row.to_dict()
    
    for i, (item_id, score) in enumerate(trending[:max_items]):
        item = item_lookup.get(item_id, {})
        title = item.get('title', f"Item #{item_id}")
        category = item.get('category', '')
        
        with cols[i]:
            html = (
                f'<div style="background: linear-gradient(180deg, rgba(99, 102, 241, 0.2) 0%, rgba(30, 41, 59, 0.8) 100%); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1rem; text-align: center; height: 140px;">'
                f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">&#127916;</div>'
                f'<div style="color: #F8FAFC; font-size: 0.85rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title[:20]}...</div>'
                f'<div style="color: #64748B; font-size: 0.7rem; margin-top: 0.25rem;">{category}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
