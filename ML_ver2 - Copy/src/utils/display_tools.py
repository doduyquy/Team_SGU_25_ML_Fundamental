"""
Display Tools for Enhanced Visualization
Provides helper functions for beautiful tables and consistent styling
"""

import pandas as pd
import numpy as np
import time
from typing import Optional
from IPython.display import HTML, display, Markdown

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("⚠️ tabulate not installed. Using default pandas display.")

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import theme system
try:
    from utils.theme import get_theme, set_theme
    THEME_AVAILABLE = True
except ImportError:
    THEME_AVAILABLE = False

# Color theme for consistency (fallback if theme not available)
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

PLOTLY_THEME = 'plotly_white'


def setup_notebook_style(theme: str = "light", performance_mode: bool = False):
    """
    Setup global styling for Jupyter notebooks with theme support
    Call this at the beginning of your notebook
    
    Parameters:
    -----------
    theme : str
        Theme name: 'light', 'dark', 'professional', 'vibrant', etc.
    performance_mode : bool
        Enable performance optimizations (disable animations, reduce effects)
    """
    # Pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:,.3f}'.format)
    pd.set_option('display.precision', 3)
    
    # Apply theme if available
    if THEME_AVAILABLE:
        set_theme(theme, performance_mode)
        theme_obj = get_theme()
        bg_color = theme_obj.config['background_color']
        text_color = theme_obj.config['text_color']
        theme_name = theme_obj.config['name']
    else:
        bg_color = '#ffffff' if theme == 'light' else '#1e1e1e'
        text_color = '#2c3e50' if theme == 'light' else '#e1e4e8'
        theme_name = theme.capitalize()
    
    # Wider notebook display with theme colors
    display(HTML(f"""
    <style>
        :root {{
            --bg-color: {bg_color};
            --text-color: {text_color};
        }}
        .container {{ width:95% !important; }}
        .output_png {{ display: table-cell; text-align: center; vertical-align: middle; }}
        div.output_scroll {{ height: 600px; }}
        body {{ background-color: var(--bg-color); color: var(--text-color); }}
        div.output_area {{ background-color: var(--bg-color); }}
        .rendered_html {{ color: var(--text-color); }}
        
        /* Better code cell styling */
        div.input_area {{
            border-radius: 5px;
            border-left: 3px solid #1f77b4;
        }}
        
        /* Prettier tables */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }}
        
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
    </style>
    """))
    
    # Display banner
    icon = "🌙" if theme == "dark" else "☀️"
    perf_tag = " [⚡ Performance Mode]" if performance_mode else ""
    display(Markdown(f"""
---
{icon} **Theme Loaded:** {theme_name}{perf_tag}  
📊 **Template Version:** 2.1.0 Enhanced
    
✨ All visualizations are now interactive! Hover, zoom, and explore.
---
    """))
    
    print("✅ Notebook styling configured successfully!")


def show_table(df, title="📋 Bảng dữ liệu", n=10, tablefmt='fancy_grid', show_index=True):
    """
    Display a beautifully formatted table
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to display
    title : str
        Title of the table
    n : int
        Number of rows to show (default: 10, -1 for all)
    tablefmt : str
        Format style: 'fancy_grid', 'github', 'pretty', 'html'
    show_index : bool
        Whether to show index
    """
    if not TABULATE_AVAILABLE:
        print(f"\n{title}")
        print("=" * 80)
        display(df.head(n) if n > 0 else df)
        return
    
    print(f"\n{title}")
    print("=" * 80)
    
    if n > 0:
        data = df.head(n)
    else:
        data = df
    
    if tablefmt == 'html':
        # For HTML display in Jupyter
        styled_df = data.style.set_properties(**{
            'background-color': '#f8f9fa',
            'border': '1px solid #dee2e6',
            'padding': '8px'
        }).set_table_styles([
            {'selector': 'thead', 'props': [('background-color', '#343a40'), ('color', 'white')]},
            {'selector': 'th', 'props': [('text-align', 'center')]},
        ])
        display(styled_df)
    else:
        # For console display
        print(tabulate(data, headers='keys', tablefmt=tablefmt, 
                      showindex=show_index, floatfmt='.3f'))
    
    print(f"\n📊 Showing {len(data)} of {len(df)} rows")


def show_metrics(metrics_dict, title="📈 Model Metrics", as_dataframe=False):
    """
    Display metrics in a beautiful format
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics {metric_name: value}
    title : str
        Title to display
    as_dataframe : bool
        If True, return as styled DataFrame
    """
    print(f"\n{title}")
    print("=" * 80)
    
    df = pd.DataFrame(metrics_dict, index=['Value']).T
    df.columns = ['Score']
    
    if TABULATE_AVAILABLE and not as_dataframe:
        print(tabulate(df, headers=['Metric', 'Score'], tablefmt='fancy_grid', floatfmt='.4f'))
    else:
        # Styled display for Jupyter
        styled_df = df.style.format('{:.4f}').background_gradient(
            cmap='RdYlGn', subset=['Score']
        ).set_properties(**{
            'font-weight': 'bold',
            'border': '1px solid #dee2e6',
            'padding': '10px'
        })
        display(styled_df)
    
    print("=" * 80)
    
    if as_dataframe:
        return df


def print_header(text, level=1, emoji="🎯"):
    """
    Print a formatted header
    
    Parameters:
    -----------
    text : str
        Header text
    level : int
        Header level (1-3)
    emoji : str
        Emoji to prepend
    """
    symbols = {1: "=", 2: "-", 3: "·"}
    lengths = {1: 100, 2: 80, 3: 60}
    
    symbol = symbols.get(level, "=")
    length = lengths.get(level, 80)
    
    print(f"\n{symbol * length}")
    print(f"{emoji} {text.upper()}" if level == 1 else f"{emoji} {text}")
    print(f"{symbol * length}\n")


def print_info(message, type='info'):
    """
    Print formatted info message
    
    Parameters:
    -----------
    message : str
        Message to print
    type : str
        Type of message: 'info', 'success', 'warning', 'error'
    """
    emojis = {
        'info': 'ℹ️',
        'success': '✅',
        'warning': '⚠️',
        'error': '❌',
        'tip': '💡'
    }
    
    emoji = emojis.get(type, 'ℹ️')
    print(f"{emoji} {message}")


def create_styled_dataframe(df, cmap='viridis', subset=None):
    """
    Create a beautifully styled DataFrame for Jupyter display
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to style
    cmap : str
        Colormap name
    subset : list
        Columns to apply gradient to
    
    Returns:
    --------
    Styled DataFrame
    """
    styled = df.style
    
    # Apply background gradient to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if subset:
        numeric_cols = [col for col in subset if col in numeric_cols]
    
    if numeric_cols:
        styled = styled.background_gradient(cmap=cmap, subset=numeric_cols)
    
    # Format floats
    styled = styled.format(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else x)
    
    # Add borders and padding
    styled = styled.set_properties(**{
        'border': '1px solid #dee2e6',
        'padding': '8px',
        'text-align': 'center'
    })
    
    # Style headers
    styled = styled.set_table_styles([
        {'selector': 'thead th', 'props': [
            ('background-color', '#343a40'),
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center')
        ]},
        {'selector': 'tbody tr:hover', 'props': [
            ('background-color', '#f8f9fa')
        ]}
    ])
    
    return styled


def get_plotly_layout(title, xaxis_title=None, yaxis_title=None, height=500, width=None):
    """
    Get consistent plotly layout configuration
    
    Parameters:
    -----------
    title : str
        Plot title
    xaxis_title : str
        X-axis label
    yaxis_title : str
        Y-axis label
    height : int
        Plot height
    width : int
        Plot width
    
    Returns:
    --------
    dict : Layout configuration
    """
    # Use theme layout if available
    if THEME_AVAILABLE:
        theme = get_theme()
        return theme.get_plotly_layout(title, height=height, width=width,
                                      xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    
    # Fallback
    layout = {
        'template': PLOTLY_THEME,
        'title': {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'weight': 'bold'}
        },
        'height': height,
        'hovermode': 'closest',
        'showlegend': True
    }
    
    if width:
        layout['width'] = width
    
    if xaxis_title:
        layout['xaxis'] = {'title': xaxis_title}
    
    if yaxis_title:
        layout['yaxis'] = {'title': yaxis_title}
    
    return layout


# ═══════════════════════════════════════════════════════════════════
# PERFORMANCE OPTIMIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def lazy_show(fig, delay: float = 0.1, config: Optional[dict] = None):
    """
    Show Plotly figure with optional delay for better performance
    Useful when rendering multiple figures sequentially
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to display
    delay : float
        Delay in seconds before showing (prevents notebook lag)
    config : dict, optional
        Plotly config options
    
    Example:
    --------
    for col in columns:
        fig = px.histogram(df, x=col)
        lazy_show(fig, delay=0.1)
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available")
        return
    
    # Default config for better performance
    if config is None:
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
        }
    
    # Add delay if specified
    if delay > 0:
        time.sleep(delay)
    
    # Show figure
    fig.show(config=config)


def show_minimal(fig, config: Optional[dict] = None):
    """
    Show Plotly figure with minimal UI for better performance
    Removes mode bar and other UI elements
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to display
    config : dict, optional
        Additional config options
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available")
        return
    
    minimal_config = {
        'displayModeBar': False,
        'staticPlot': False,
        'displaylogo': False,
    }
    
    if config:
        minimal_config.update(config)
    
    fig.show(config=minimal_config)


def optimize_figure_for_performance(fig, disable_animation: bool = True, 
                                   reduce_points: bool = False,
                                   max_points: int = 1000):
    """
    Optimize Plotly figure for better performance
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to optimize
    disable_animation : bool
        Disable animations
    reduce_points : bool
        Reduce number of data points if too many
    max_points : int
        Maximum number of points to display
    
    Returns:
    --------
    fig : Optimized figure
    """
    if not PLOTLY_AVAILABLE:
        return fig
    
    # Disable animations
    if disable_animation:
        fig.layout.transition = {'duration': 0}
        fig.layout.updatemenus = None
    
    # Reduce data points if needed
    if reduce_points:
        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                if len(trace.x) > max_points:
                    # Sample data
                    indices = np.linspace(0, len(trace.x) - 1, max_points, dtype=int)
                    trace.x = [trace.x[i] for i in indices]
                    if hasattr(trace, 'y') and trace.y is not None:
                        trace.y = [trace.y[i] for i in indices]
    
    return fig


def batch_show_figures(figures: list, delay: float = 0.15, 
                      performance_mode: bool = True):
    """
    Show multiple figures with optimizations for better notebook performance
    
    Parameters:
    -----------
    figures : list
        List of Plotly figures
    delay : float
        Delay between figures
    performance_mode : bool
        Apply performance optimizations
    
    Example:
    --------
    figs = [fig1, fig2, fig3]
    batch_show_figures(figs, delay=0.2)
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available")
        return
    
    config = {
        'displayModeBar': not performance_mode,
        'displaylogo': False,
    }
    
    for i, fig in enumerate(figures):
        if performance_mode:
            fig = optimize_figure_for_performance(fig)
        
        print(f"📊 Figure {i+1}/{len(figures)}")
        lazy_show(fig, delay=delay, config=config)


def set_plotly_renderer(renderer: str = "notebook_connected"):
    """
    Set Plotly renderer for better performance
    
    Parameters:
    -----------
    renderer : str
        Renderer name: 'notebook', 'notebook_connected', 'jupyterlab', etc.
        'notebook_connected' is recommended for better performance
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not available")
        return
    
    pio.renderers.default = renderer
    print(f"✅ Plotly renderer set to: {renderer}")


# ═══════════════════════════════════════════════════════════════════
# NAVIGATION AND UX HELPERS
# ═══════════════════════════════════════════════════════════════════

def create_navigation_menu(sections: dict):
    """
    Create a navigation menu for Jupyter notebooks
    
    Parameters:
    -----------
    sections : dict
        Dictionary of section names and their cell numbers
        Example: {'Introduction': 1, 'EDA': 5, 'Modeling': 10}
    
    Example:
    --------
    create_navigation_menu({
        '📊 Data Loading': 2,
        '🔍 EDA': 5,
        '🤖 Modeling': 10,
        '📈 Evaluation': 15
    })
    """
    html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h2 style="color: white; margin: 0 0 15px 0;">📑 Quick Navigation</h2>
        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    """
    
    for section, cell_num in sections.items():
        html += f"""
        <a href="#" onclick="Jupyter.notebook.scroll_to_cell({cell_num}); return false;"
           style="background: white; color: #667eea; padding: 10px 15px; 
                  border-radius: 5px; text-decoration: none; font-weight: bold;
                  box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
            {section}
        </a>
        """
    
    html += """
        </div>
    </div>
    """
    
    display(HTML(html))


def show_progress_indicator(current: int, total: int, message: str = "Processing"):
    """
    Show a simple progress indicator
    
    Parameters:
    -----------
    current : int
        Current progress
    total : int
        Total items
    message : str
        Progress message
    """
    percentage = (current / total) * 100
    bar_length = 40
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\r{message}: [{bar}] {percentage:.1f}% ({current}/{total})", end='', flush=True)
    
    if current == total:
        print()  # New line when complete

