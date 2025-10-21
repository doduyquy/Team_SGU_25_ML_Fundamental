# Utils package - Enhanced Display Tools & Theme System

from .display_tools import (
    setup_notebook_style,
    show_table,
    show_metrics,
    print_header,
    print_info,
    create_styled_dataframe,
    get_plotly_layout,
    lazy_show,
    show_minimal,
    optimize_figure_for_performance,
    batch_show_figures,
    set_plotly_renderer,
    create_navigation_menu,
    show_progress_indicator,
    COLORS,
    PLOTLY_THEME
)

from .theme import (
    VisualizationTheme,
    get_theme,
    set_theme,
    switch_theme,
    list_themes
)

__all__ = [
    # Display tools
    'setup_notebook_style',
    'show_table',
    'show_metrics',
    'print_header',
    'print_info',
    'create_styled_dataframe',
    'get_plotly_layout',
    # Performance tools
    'lazy_show',
    'show_minimal',
    'optimize_figure_for_performance',
    'batch_show_figures',
    'set_plotly_renderer',
    # Navigation tools
    'create_navigation_menu',
    'show_progress_indicator',
    # Theme system
    'VisualizationTheme',
    'get_theme',
    'set_theme',
    'switch_theme',
    'list_themes',
    # Constants
    'COLORS',
    'PLOTLY_THEME'
]
