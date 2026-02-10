# Metrics Visualization Component
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd


def create_plotly_theme() -> Dict:
    """Create consistent dark theme for Plotly charts."""
    return {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#94A3B8', 'family': 'Inter, sans-serif'},
        'xaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'zerolinecolor': 'rgba(148, 163, 184, 0.2)'
        },
        'yaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'zerolinecolor': 'rgba(148, 163, 184, 0.2)'
        }
    }


def render_metrics_dashboard(metrics: Dict[str, float]) -> None:
    """Render comprehensive metrics dashboard."""
    
    st.markdown("### Model Performance Metrics")
    
    # Key metrics cards
    key_metrics = [
        ('precision@10', 'Precision@10', '&#127919;'),
        ('recall@10', 'Recall@10', '&#128269;'),
        ('ndcg@10', 'NDCG@10', '&#128200;'),
        ('map', 'MAP', '&#128506;'),
        ('coverage', 'Coverage', '&#128230;'),
        ('diversity', 'Diversity', '&#127752;')
    ]
    
    cols = st.columns(3)
    
    for i, (key, label, icon) in enumerate(key_metrics):
        value = metrics.get(key, 0)
        
        with cols[i % 3]:
            # Determine color based on value
            if value >= 0.3:
                color = '#10B981'  # Green
            elif value >= 0.15:
                color = '#F59E0B'  # Yellow
            else:
                color = '#EF4444'  # Red
            
            bar_width = min(100, value * 100)
            
            html = (
                f'<div style="background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;">'
                f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                f'<span style="font-size: 1.5rem;">{icon}</span>'
                f'<span style="font-size: 1.75rem; font-weight: 800; color: {color};">{value:.1%}</span>'
                f'</div>'
                f'<div style="color: #94A3B8; font-size: 0.85rem; margin-top: 0.5rem;">{label}</div>'
                f'<div style="margin-top: 0.75rem; height: 4px; background: rgba(51, 65, 85, 0.8); border-radius: 9999px; overflow: hidden;">'
                f'<div style="width: {bar_width}%; height: 100%; background: linear-gradient(90deg, {color}, {color}88); border-radius: 9999px;"></div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)


def render_comparison_chart(comparison_data: Dict) -> None:
    """Render model comparison radar chart."""
    
    st.markdown("### Algorithm Comparison")
    
    metrics = comparison_data.get('metrics', {})
    models = comparison_data.get('model_names', [])
    
    if not models or not metrics:
        st.info("Run model comparison to see results.")
        return
    
    # Create radar chart
    categories = ['Precision@10', 'Recall@10', 'NDCG@10', 'MAP', 'Coverage']
    metric_keys = ['precision@10', 'recall@10', 'ndcg@10', 'map', 'coverage']
    
    colors = ['#6366F1', '#8B5CF6', '#06B6D4', '#10B981', '#F59E0B']
    
    fig = go.Figure()
    
    for i, model in enumerate(models):
        model_metrics = metrics.get(model, {})
        values = [model_metrics.get(k, 0) for k in metric_keys]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=model,
            line_color=colors[i % len(colors)],
            fillcolor=colors[i % len(colors)] + '33'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(0.5, max(max(metrics.get(m, {}).get(k, 0) for k in metric_keys) for m in models) * 1.2)],
                gridcolor='rgba(148, 163, 184, 0.1)',
                linecolor='rgba(148, 163, 184, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(148, 163, 184, 0.1)',
                linecolor='rgba(148, 163, 184, 0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(color='#94A3B8')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94A3B8'),
        margin=dict(t=30, b=80, l=50, r=50),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart comparison
    st.markdown("#### Detailed Comparison")
    
    df_data = []
    for model in models:
        model_metrics = metrics.get(model, {})
        for metric_key, metric_name in zip(metric_keys, categories):
            df_data.append({
                'Model': model,
                'Metric': metric_name,
                'Value': model_metrics.get(metric_key, 0)
            })
    
    df = pd.DataFrame(df_data)
    
    fig_bar = px.bar(
        df,
        x='Metric',
        y='Value',
        color='Model',
        barmode='group',
        color_discrete_sequence=colors
    )
    
    fig_bar.update_layout(
        **create_plotly_theme(),
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        )
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)


def render_interaction_distribution(interactions_df) -> None:
    """Render interaction type distribution chart."""
    
    if interactions_df is None or len(interactions_df) == 0:
        return
    
    st.markdown("### Interaction Distribution")
    
    # Interaction type counts
    type_counts = interactions_df['interaction_type'].value_counts()
    
    colors = {
        'view': '#3B82F6',
        'click': '#8B5CF6',
        'like': '#EC4899',
        'purchase': '#10B981'
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=type_counts.index.tolist(),
            values=type_counts.values.tolist(),
            hole=0.6,
            marker_colors=[colors.get(t, '#6366F1') for t in type_counts.index],
            textinfo='percent+label',
            textposition='outside',
            textfont=dict(color='#F8FAFC')
        )
    ])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94A3B8'),
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        annotations=[dict(
            text=f'{len(interactions_df):,}<br>Total',
            x=0.5, y=0.5,
            font_size=16,
            font_color='#F8FAFC',
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_category_popularity(items_df, interactions_df) -> None:
    """Render category popularity treemap."""
    
    if items_df is None or interactions_df is None:
        return
    
    st.markdown("### Category Popularity")
    
    # Join interactions with items
    item_cats = dict(zip(items_df['item_id'], items_df['category']))
    interactions_df = interactions_df.copy()
    interactions_df['category'] = interactions_df['item_id'].map(item_cats)
    
    cat_counts = interactions_df['category'].value_counts()
    
    fig = go.Figure(data=[
        go.Bar(
            x=cat_counts.values[:10],
            y=cat_counts.index[:10],
            orientation='h',
            marker=dict(
                color=cat_counts.values[:10],
                colorscale=[[0, '#6366F1'], [1, '#8B5CF6']],
            )
        )
    ])
    
    fig.update_layout(
        **create_plotly_theme(),
        height=300,
        margin=dict(t=20, b=40, l=100, r=20),
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)
