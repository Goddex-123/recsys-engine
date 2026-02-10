# Live Simulation Component
import streamlit as st
from typing import Dict, List, Tuple, Any, Callable
import random


def render_live_simulation(
    items_df,
    current_user_id: int,
    on_interaction: Callable = None,
    session_state_key: str = 'simulation'
) -> Dict:
    """Render interactive live simulation panel."""
    
    st.markdown("### Live Interaction Simulation")
    
    html = (
        '<div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1)); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">'
        '<p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">Interact with items below to see how recommendations update in real-time!</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    
    # Initialize session state for simulation
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = {
            'interactions': [],
            'last_update': None
        }
    
    sim_state = st.session_state[session_state_key]
    
    # Random items for interaction
    if items_df is not None and len(items_df) > 0:
        sample_items = items_df.sample(min(4, len(items_df))).to_dict('records')
    else:
        sample_items = []
    
    if sample_items:
        st.markdown("**Select an item to interact with:**")
        
        cols = st.columns(4)
        
        for i, item in enumerate(sample_items):
            with cols[i]:
                item_id = item.get('item_id')
                title = item.get('title', f"Item {item_id}")[:25] + "..."
                category = item.get('category', '')
                
                html = (
                    f'<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 10px; padding: 0.75rem; text-align: center; margin-bottom: 0.5rem; min-height: 100px;">'
                    f'<div style="font-size: 1.5rem; margin-bottom: 0.25rem;">&#127916;</div>'
                    f'<div style="color: #F8FAFC; font-size: 0.8rem; font-weight: 500;">{title}</div>'
                    f'<div style="color: #64748B; font-size: 0.7rem;">{category}</div>'
                    f'</div>'
                )
                st.markdown(html, unsafe_allow_html=True)
                
                # Interaction buttons
                col_btns = st.columns(2)
                with col_btns[0]:
                    if st.button("View", key=f"view_{item_id}", help="View"):
                        sim_state['interactions'].append({
                            'item_id': item_id,
                            'action': 'view',
                            'title': title
                        })
                        st.rerun()
                
                with col_btns[1]:
                    if st.button("Like", key=f"like_{item_id}", help="Like"):
                        sim_state['interactions'].append({
                            'item_id': item_id,
                            'action': 'like',
                            'title': title
                        })
                        st.rerun()
    
    # Show recent interactions
    if sim_state['interactions']:
        st.markdown("---")
        st.markdown("**Session Activity:**")
        
        for interaction in sim_state['interactions'][-5:]:
            icon = '&#128065;' if interaction['action'] == 'view' else '&#10084;&#65039;'
            html = (
                f'<div style="display: inline-flex; align-items: center; background: rgba(99, 102, 241, 0.1); padding: 0.25rem 0.75rem; border-radius: 9999px; margin: 0.25rem;">'
                f'<span style="margin-right: 0.25rem;">{icon}</span>'
                f'<span style="color: #A5B4FC; font-size: 0.8rem;">{interaction["title"]}</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
        
        if st.button("Clear Session", key="clear_sim"):
            sim_state['interactions'] = []
            st.rerun()
    
    return sim_state


def render_cold_start_demo(
    models: Dict[str, Any],
    items_df
) -> None:
    """Demonstrate cold-start handling."""
    
    st.markdown("### Cold-Start User Demo")
    
    html = (
        '<div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 1rem; margin-bottom: 1rem;">'
        '<p style="color: #FBBF24; margin: 0 0 0.5rem 0; font-weight: 600;">Cold-Start Challenge</p>'
        '<p style="color: #94A3B8; margin: 0; font-size: 0.9rem;">New users have no interaction history. See how different algorithms handle this challenge.</p>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
    
    # Simulate new user
    new_user_id = -999  # Fake new user
    
    st.markdown("**Recommendations for a brand new user:**")
    
    if models:
        tabs = st.tabs(list(models.keys()))
        
        for tab, (model_name, model) in zip(tabs, models.items()):
            with tab:
                try:
                    recs = model.recommend(new_user_id, n=5, exclude_seen=False)
                    
                    if recs:
                        st.success(f"{model_name} provided {len(recs)} recommendations")
                        
                        for item_id, score in recs[:3]:
                            item_title = "Unknown"
                            if items_df is not None:
                                item_row = items_df[items_df['item_id'] == item_id]
                                if len(item_row) > 0:
                                    item_title = item_row.iloc[0]['title']
                            
                            st.markdown(f"- {item_title} (score: {score:.3f})")
                    else:
                        st.warning(f"{model_name} returned no recommendations for cold-start user")
                        
                except Exception as e:
                    st.error(f"Error: {str(e)[:50]}")
    
    # Strategy explanation
    with st.expander("Cold-Start Strategies Explained"):
        st.markdown("""
        **How each model handles cold-start:**
        
        | Model | Strategy | Effectiveness |
        |-------|----------|---------------|
        | **Popularity** | Recommends globally popular items | Good fallback |
        | **User-CF** | Cannot recommend (no similar users) | Fails |
        | **Item-CF** | Cannot recommend (no item history) | Fails |
        | **SVD** | Uses global biases only | Limited |
        | **Hybrid** | Falls back to content + popularity | Best handling |
        
        The **Hybrid** model is designed to gracefully handle cold-start by 
        automatically switching strategies based on user history.
        """)


def render_ab_test_simulation() -> None:
    """Simulate A/B testing concept."""
    
    st.markdown("### A/B Test Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        html = (
            '<div style="background: rgba(99, 102, 241, 0.1); border: 2px solid rgba(99, 102, 241, 0.5); border-radius: 12px; padding: 1.25rem; text-align: center;">'
            '<h4 style="color: #A5B4FC; margin: 0 0 0.5rem 0;">Control (A)</h4>'
            '<div style="font-size: 2rem; margin: 0.5rem 0;">&#128202;</div>'
            '<p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Popularity-Based</p>'
            '<div style="margin-top: 1rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.2); border-radius: 8px;">'
            '<span style="color: #6EE7B7; font-size: 1.25rem; font-weight: 700;">12.4%</span>'
            '<span style="color: #94A3B8; font-size: 0.8rem;"> CTR</span>'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    
    with col2:
        html = (
            '<div style="background: rgba(139, 92, 246, 0.1); border: 2px solid rgba(139, 92, 246, 0.5); border-radius: 12px; padding: 1.25rem; text-align: center;">'
            '<h4 style="color: #C4B5FD; margin: 0 0 0.5rem 0;">Treatment (B)</h4>'
            '<div style="font-size: 2rem; margin: 0.5rem 0;">&#127919;</div>'
            '<p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Hybrid Model</p>'
            '<div style="margin-top: 1rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.2); border-radius: 8px;">'
            '<span style="color: #6EE7B7; font-size: 1.25rem; font-weight: 700;">18.7%</span>'
            '<span style="color: #94A3B8; font-size: 0.8rem;"> CTR</span>'
            '</div>'
            '</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    
    html = (
        '<div style="text-align: center; margin-top: 1rem; padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px;">'
        '<span style="color: #6EE7B7; font-weight: 600;">&uarr; 50.8% improvement with Hybrid model (p &lt; 0.001)</span>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)
