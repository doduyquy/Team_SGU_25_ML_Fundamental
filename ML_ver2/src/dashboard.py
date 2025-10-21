"""
Dashboard Module for Comprehensive EDA & Model Evaluation
Provides unified dashboard views with multiple visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not installed. Dashboard features limited.")

try:
    from utils.display_tools import print_header, print_info
    DISPLAY_TOOLS_AVAILABLE = True
except ImportError:
    DISPLAY_TOOLS_AVAILABLE = False
    def print_header(text, **kwargs):
        print(f"\n{'='*80}\n{text}\n{'='*80}")
    def print_info(msg, **kwargs):
        print(msg)


def show_overview_dashboard(df, target_col=None, max_features=10):
    """
    Hiển thị dashboard tổng quan cho EDA
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame cần phân tích
    target_col : str
        Tên cột target (optional)
    max_features : int
        Số lượng features tối đa hiển thị
    """
    if not PLOTLY_AVAILABLE:
        print_info("⚠️ Plotly không khả dụng. Dashboard bị giới hạn.", type='warning')
        return
    
    print_header("📊 DASHBOARD TỔNG QUAN", level=1, emoji="📊")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Limit features
    numeric_cols = numeric_cols[:max_features]
    
    print_info(f"Hiển thị {len(numeric_cols)} features", type='info')
    
    # Create subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f'Distribution: {col}' for col in numeric_cols],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Add histograms
    for idx, col in enumerate(numeric_cols):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                nbinsx=30,
                showlegend=False,
                marker=dict(color='lightblue', line=dict(color='darkblue', width=1))
            ),
            row=row,
            col=col_pos
        )
    
    fig.update_layout(
        title_text='📊 Distributions of Numeric Features',
        template='plotly_white',
        height=300 * n_rows,
        showlegend=False
    )
    
    fig.show()


def show_correlation_dashboard(df, method='pearson', threshold=0.5):
    """
    Dashboard hiển thị correlation matrix và top correlations
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    method : str
        'pearson', 'spearman', or 'kendall'
    threshold : float
        Threshold để hiển thị top correlations
    """
    if not PLOTLY_AVAILABLE:
        print_info("⚠️ Plotly không khả dụng.", type='warning')
        return
    
    print_header("🔗 CORRELATION DASHBOARD", level=1, emoji="🔗")
    
    # Calculate correlation
    corr = df.corr(method=method, numeric_only=True)
    
    # Create figure with 2 subplots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=('Correlation Heatmap', 'Top Correlations'),
        column_widths=[0.6, 0.4],
        specs=[[{"type": "heatmap"}, {"type": "bar"}]]
    )
    
    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Correlation", x=0.45)
        ),
        row=1,
        col=1
    )
    
    # Top correlations (excluding diagonal)
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_pairs.append({
                'pair': f'{corr.columns[i]} - {corr.columns[j]}',
                'value': abs(corr.iloc[i, j])
            })
    
    corr_df = pd.DataFrame(corr_pairs).sort_values('value', ascending=False).head(15)
    
    fig.add_trace(
        go.Bar(
            x=corr_df['value'],
            y=corr_df['pair'],
            orientation='h',
            marker=dict(color=corr_df['value'], colorscale='Viridis'),
            showlegend=False
        ),
        row=1,
        col=2
    )
    
    fig.update_xaxes(title_text="Features", row=1, col=1)
    fig.update_yaxes(title_text="Features", row=1, col=1, autorange='reversed')
    fig.update_xaxes(title_text="Correlation (abs)", row=1, col=2)
    
    fig.update_layout(
        title_text=f'🔗 Correlation Analysis ({method})',
        template='plotly_white',
        height=600,
        width=1400
    )
    
    fig.show()


def show_model_comparison_dashboard(models_metrics):
    """
    Dashboard so sánh nhiều models
    
    Parameters:
    -----------
    models_metrics : dict
        Dictionary {model_name: metrics_dict}
        Example: {'Lasso': {'MAE': 100, 'RMSE': 150, 'R²': 0.85}, ...}
    """
    if not PLOTLY_AVAILABLE:
        print_info("⚠️ Plotly không khả dụng.", type='warning')
        return
    
    print_header("🏆 MODEL COMPARISON DASHBOARD", level=1, emoji="🏆")
    
    # Convert to DataFrame
    df = pd.DataFrame(models_metrics).T
    
    # Get metric names
    metrics = df.columns.tolist()
    n_metrics = len(metrics)
    
    # Create subplots
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=metrics,
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Add bar charts for each metric
    for idx, metric in enumerate(metrics):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df[metric],
                name=metric,
                marker=dict(color=df[metric], colorscale='Viridis'),
                showlegend=False,
                text=df[metric].round(4),
                textposition='auto'
            ),
            row=row,
            col=col_pos
        )
        
        fig.update_yaxes(title_text=metric, row=row, col=col_pos)
    
    fig.update_layout(
        title_text='🏆 Model Metrics Comparison',
        template='plotly_white',
        height=400 * n_rows,
        showlegend=False
    )
    
    fig.show()
    
    # Show table
    print_info("\n📋 Detailed Metrics Table:", type='info')
    if DISPLAY_TOOLS_AVAILABLE:
        from utils.display_tools import show_table
        show_table(df, title="Model Comparison", n=-1, tablefmt='fancy_grid')
    else:
        print(df)


def show_target_analysis_dashboard(df, target_col):
    """
    Dashboard phân tích biến target
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    target_col : str
        Tên cột target
    """
    if not PLOTLY_AVAILABLE:
        print_info("⚠️ Plotly không khả dụng.", type='warning')
        return
    
    if target_col not in df.columns:
        print_info(f"⚠️ Column '{target_col}' không tồn tại!", type='error')
        return
    
    print_header(f"🎯 TARGET ANALYSIS: {target_col}", level=1, emoji="🎯")
    
    # Determine if target is numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    
    if is_numeric:
        # Create subplots for numeric target
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Distribution',
                'Box Plot',
                'Correlation with Features',
                'Statistics'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=df[target_col],
                nbinsx=30,
                name='Distribution',
                marker=dict(color='lightblue'),
                showlegend=False
            ),
            row=1,
            col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=df[target_col],
                name='Box Plot',
                marker=dict(color='lightgreen'),
                showlegend=False
            ),
            row=1,
            col=2
        )
        
        # Correlation with other features
        corr_with_target = df.corr(numeric_only=True)[target_col].abs().sort_values(ascending=False)[1:11]
        
        fig.add_trace(
            go.Bar(
                x=corr_with_target.values,
                y=corr_with_target.index,
                orientation='h',
                name='Correlation',
                marker=dict(color=corr_with_target.values, colorscale='Viridis'),
                showlegend=False
            ),
            row=2,
            col=1
        )
        
        # Statistics table
        stats = df[target_col].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(
                    values=[
                        stats.index.tolist(),
                        [f'{v:.2f}' for v in stats.values]
                    ]
                )
            ),
            row=2,
            col=2
        )
        
        fig.update_xaxes(title_text=target_col, row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Correlation", row=2, col=1)
        
    else:
        # Categorical target
        value_counts = df[target_col].value_counts()
        
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=('Distribution', 'Proportions'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                marker=dict(color=value_counts.values, colorscale='Viridis'),
                showlegend=False
            ),
            row=1,
            col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                showlegend=True
            ),
            row=1,
            col=2
        )
        
        fig.update_xaxes(title_text=target_col, row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_layout(
        title_text=f'🎯 Target Analysis: {target_col}',
        template='plotly_white',
        height=700,
        showlegend=False
    )
    
    fig.show()


def show_feature_target_relationship(df, features, target_col, plot_type='auto'):
    """
    Hiển thị mối quan hệ giữa features và target
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame
    features : list
        Danh sách features cần phân tích
    target_col : str
        Tên cột target
    plot_type : str
        'auto', 'scatter', 'box', or 'violin'
    """
    if not PLOTLY_AVAILABLE:
        print_info("⚠️ Plotly không khả dụng.", type='warning')
        return
    
    print_header("🔍 FEATURE-TARGET RELATIONSHIP", level=1, emoji="🔍")
    
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f'{f} vs {target_col}' for f in features],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, feature in enumerate(features):
        if feature not in df.columns:
            continue
        
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1
        
        # Determine plot type
        is_numeric_feature = pd.api.types.is_numeric_dtype(df[feature])
        is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
        
        if is_numeric_feature and is_numeric_target:
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=df[feature],
                    y=df[target_col],
                    mode='markers',
                    marker=dict(size=5, opacity=0.6),
                    showlegend=False
                ),
                row=row,
                col=col_pos
            )
        else:
            # Box plot for categorical
            fig.add_trace(
                go.Box(
                    x=df[feature],
                    y=df[target_col],
                    showlegend=False
                ),
                row=row,
                col=col_pos
            )
        
        fig.update_xaxes(title_text=feature, row=row, col=col_pos)
        fig.update_yaxes(title_text=target_col, row=row, col=col_pos)
    
    fig.update_layout(
        title_text=f'🔍 Features vs {target_col}',
        template='plotly_white',
        height=400 * n_rows,
        showlegend=False
    )
    
    fig.show()


# Quick access function
def quick_eda_dashboard(df, target_col=None):
    """
    Quick comprehensive EDA dashboard
    Combines overview, correlations, and target analysis
    """
    print_header("⚡ QUICK EDA DASHBOARD", level=1, emoji="⚡")
    
    print_info(f"Dataset shape: {df.shape}", type='info')
    print_info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB", type='info')
    
    # Overview
    show_overview_dashboard(df, target_col=target_col, max_features=9)
    
    # Correlation
    show_correlation_dashboard(df, method='pearson')
    
    # Target analysis
    if target_col:
        show_target_analysis_dashboard(df, target_col)
    
    print_header("✅ DASHBOARD COMPLETE", level=1, emoji="✅")

