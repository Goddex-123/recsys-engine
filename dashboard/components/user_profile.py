# User Profile Component - Premium Profile Cards
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


def render_user_profile(
    user_data: Dict,
    interaction_count: int,
    preferred_categories: List[str]
) -> None:
    """Render premium user profile card."""
    
    # Profile header with avatar
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Avatar circle with initials
        initials = user_data.get('username', 'U')[:2].upper()
        html = (
            f'<div style="width: 80px; height: 80px; border-radius: 50%; background: linear-gradient(135deg, #6366F1, #8B5CF6); display: flex; align-items: center; justify-content: center; font-size: 28px; font-weight: 700; color: white; box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);">'
            f'{initials}'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    
    with col2:
        username = user_data.get('username', 'Unknown User')
        premium_badge = '&#11088; Premium' if user_data.get('is_premium') else '&#128100; Standard'
        html = (
            f'<div>'
            f'<h3 style="margin: 0; color: #F8FAFC; font-size: 1.5rem;">{username}</h3>'
            f'<p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">User ID: {user_data.get("user_id", "N/A")} &bull; {premium_badge}</p>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    
    # Stats row
    stat_cols = st.columns(4)
    
    stats = [
        {"label": "Interactions", "value": f"{interaction_count:,}", "icon": "&#127916;"},
        {"label": "Age Group", "value": user_data.get('age_group', 'N/A'), "icon": "&#128197;"},
        {"label": "Member Since", "value": str(user_data.get('signup_date', 'N/A'))[:10], "icon": "&#128198;"},
        {"label": "Preferences", "value": str(len(preferred_categories)), "icon": "&#10084;&#65039;"}
    ]
    
    for col, stat in zip(stat_cols, stats):
        with col:
            html = (
                f'<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 1rem; text-align: center;">'
                f'<div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{stat["icon"]}</div>'
                f'<div style="font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">{stat["value"]}</div>'
                f'<div style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">{stat["label"]}</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
    
    # Preferred categories
    if preferred_categories:
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("**Preferred Categories**")
        
        categories_html = ""
        for cat in preferred_categories:
            categories_html += (
                f'<span style="display: inline-block; background: rgba(99, 102, 241, 0.2); color: #A5B4FC; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 500; margin: 0.25rem 0.25rem 0.25rem 0;">{cat}</span>'
            )
        
        st.markdown(categories_html, unsafe_allow_html=True)


def render_user_history(
    history_df: pd.DataFrame,
    items_df: pd.DataFrame,
    max_items: int = 10
) -> None:
    """Render user's recent interaction history."""
    
    if len(history_df) == 0:
        st.info("No interaction history found for this user.")
        return
    
    st.markdown("### Recent Activity")
    
    # Join with items for titles
    history_with_titles = history_df.head(max_items).copy()
    
    if items_df is not None:
        item_titles = dict(zip(items_df['item_id'], items_df['title']))
        item_categories = dict(zip(items_df['item_id'], items_df['category']))
        history_with_titles['title'] = history_with_titles['item_id'].map(item_titles)
        history_with_titles['category'] = history_with_titles['item_id'].map(item_categories)
    
    # Action icons (using HTML entities to avoid encoding issues)
    action_icons = {
        'view': '&#128065;',
        'click': '&#128070;',
        'like': '&#10084;&#65039;',
        'purchase': '&#128176;'
    }
    
    for _, row in history_with_titles.iterrows():
        action = row.get('interaction_type', 'view')
        icon = action_icons.get(action, '&bull;')
        title = row.get('title', f"Item #{row['item_id']}")
        category = row.get('category', '')
        timestamp = str(row.get('timestamp', ''))[:16]
        
        html = (
            f'<div style="display: flex; align-items: center; padding: 0.75rem 1rem; background: rgba(30, 41, 59, 0.5); border-radius: 8px; margin-bottom: 0.5rem; border: 1px solid rgba(148, 163, 184, 0.1);">'
            f'<span style="font-size: 1.25rem; margin-right: 0.75rem;">{icon}</span>'
            f'<div style="flex: 1;">'
            f'<div style="color: #F8FAFC; font-weight: 500;">{title}</div>'
            f'<div style="color: #64748B; font-size: 0.8rem;">{category} &bull; {timestamp}</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(html, unsafe_allow_html=True)


def render_user_selector(users_df: pd.DataFrame) -> Optional[int]:
    """Render user selection dropdown with preview."""
    
    if users_df is None or len(users_df) == 0:
        st.warning("No users available")
        return None
    
    # Create display options
    options = []
    for _, row in users_df.head(500).iterrows():
        label = f"User: {row['username']} (ID: {row['user_id']})"
        if row.get('is_premium'):
            label = f"* {row['username']} (ID: {row['user_id']})"
        options.append((row['user_id'], label))
    
    selected = st.selectbox(
        "Select User",
        options=[opt[0] for opt in options],
        format_func=lambda x: next(opt[1] for opt in options if opt[0] == x),
        help="Choose a user to view their profile and recommendations"
    )
    
    return selected
