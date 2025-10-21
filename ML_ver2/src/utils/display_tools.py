"""
Display Tools for Enhanced Visualization
Provides helper functions for beautiful tables and consistent styling
"""

import pandas as pd
import numpy as np
from IPython.display import HTML, display
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("⚠️ tabulate not installed. Using default pandas display.")

# Color theme for consistency
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


def setup_notebook_style():
    """
    Setup global styling for Jupyter notebooks
    Call this at the beginning of your notebook
    """
    # Pandas display options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.float_format', '{:,.3f}'.format)
    pd.set_option('display.precision', 3)
    
    # Wider notebook display
    display(HTML("""
    <style>
        .container { width:95% !important; }
        .output_png { display: table-cell; text-align: center; vertical-align: middle; }
        div.output_scroll { height: 600px; }
    </style>
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

