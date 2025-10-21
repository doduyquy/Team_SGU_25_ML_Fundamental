"""
Theme System for Enhanced Visualization
Provides consistent theming across all visualizations
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Any

try:
    import plotly.io as pio
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class VisualizationTheme:
    """
    Centralized theme management for all visualizations
    Supports light and dark themes with consistent styling
    """
    
    # Pre-defined themes
    THEMES = {
        "light": {
            "name": "Modern Light",
            "plotly_template": "plotly_white",
            "primary_color": "#1f77b4",
            "secondary_color": "#ff7f0e",
            "accent_color": "#2ca02c",
            "background_color": "#ffffff",
            "text_color": "#2c3e50",
            "grid_color": "#e1e4e8",
            "font_family": "Inter, Segoe UI, Roboto, Arial, sans-serif",
            "font_size": 12,
            "title_font_size": 18,
            "hover_bg_color": "#f8f9fa",
            "colorscale": "Viridis",
            "hovermode": "x unified",
        },
        "dark": {
            "name": "Modern Dark",
            "plotly_template": "plotly_dark",
            "primary_color": "#00d9ff",
            "secondary_color": "#ff6b6b",
            "accent_color": "#51cf66",
            "background_color": "#1e1e1e",
            "text_color": "#e1e4e8",
            "grid_color": "#2d3748",
            "font_family": "Inter, Segoe UI, Roboto, Arial, sans-serif",
            "font_size": 12,
            "title_font_size": 18,
            "hover_bg_color": "#2d3748",
            "colorscale": "Plasma",
            "hovermode": "x unified",
        },
        "professional": {
            "name": "Professional",
            "plotly_template": "plotly_white",
            "primary_color": "#2E4053",
            "secondary_color": "#5DADE2",
            "accent_color": "#F39C12",
            "background_color": "#FDFEFE",
            "text_color": "#34495E",
            "grid_color": "#ECF0F1",
            "font_family": "Georgia, Times New Roman, serif",
            "font_size": 13,
            "title_font_size": 20,
            "hover_bg_color": "#F8F9F9",
            "colorscale": "Blues",
            "hovermode": "closest",
        },
        "vibrant": {
            "name": "Vibrant",
            "plotly_template": "plotly",
            "primary_color": "#FF6B9D",
            "secondary_color": "#C44569",
            "accent_color": "#FFA502",
            "background_color": "#FFFFFF",
            "text_color": "#2C3335",
            "grid_color": "#EAF0F1",
            "font_family": "Montserrat, Arial, sans-serif",
            "font_size": 12,
            "title_font_size": 19,
            "hover_bg_color": "#F5F6FA",
            "colorscale": "Rainbow",
            "hovermode": "x unified",
        }
    }
    
    def __init__(self, theme: str = "light", config_path: Optional[str] = None):
        """
        Initialize theme manager
        
        Parameters:
        -----------
        theme : str
            Theme name: 'light', 'dark', 'professional', 'vibrant'
        config_path : str, optional
            Path to custom theme config JSON
        """
        self.theme_name = theme
        self._load_theme(theme, config_path)
        self._current_theme = theme
    
    def _load_theme(self, theme: str, config_path: Optional[str] = None):
        """Load theme configuration"""
        # Load from custom config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_themes = json.load(f)
                if theme in custom_themes:
                    self.config = custom_themes[theme]
                    return
        
        # Load from predefined themes
        if theme in self.THEMES:
            self.config = self.THEMES[theme].copy()
        else:
            print(f"⚠️ Theme '{theme}' not found. Using 'light' theme.")
            self.config = self.THEMES["light"].copy()
    
    def apply_theme(self, performance_mode: bool = False):
        """
        Apply theme to all visualization libraries
        
        Parameters:
        -----------
        performance_mode : bool
            If True, disable animations and reduce effects for better performance
        """
        # Apply to Plotly
        if PLOTLY_AVAILABLE:
            self._apply_plotly_theme(performance_mode)
        
        # Apply to Matplotlib/Seaborn
        if MATPLOTLIB_AVAILABLE:
            self._apply_matplotlib_theme()
        
        # Print confirmation
        icon = "🌙" if "dark" in self.theme_name.lower() else "☀️"
        perf_tag = " [⚡ Performance Mode]" if performance_mode else ""
        print(f"{icon} Theme Loaded: {self.config['name']}{perf_tag}")
    
    def _apply_plotly_theme(self, performance_mode: bool = False):
        """Apply theme to Plotly"""
        # Set default template
        pio.templates.default = self.config['plotly_template']
        
        # Set default renderer for better performance
        if performance_mode:
            pio.renderers.default = "notebook_connected"
        
        # Create custom template
        custom_template = go.layout.Template()
        
        # Layout defaults
        layout_config = {
            'font': {
                'family': self.config['font_family'],
                'size': self.config['font_size'],
                'color': self.config['text_color']
            },
            'title': {
                'font': {
                    'size': self.config['title_font_size'],
                    'family': self.config['font_family']
                }
            },
            'hovermode': self.config['hovermode'],
            'hoverlabel': {
                'bgcolor': self.config['hover_bg_color'],
                'font_size': 13,
                'font_family': self.config['font_family']
            },
            'colorway': [
                self.config['primary_color'],
                self.config['secondary_color'],
                self.config['accent_color'],
                '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ],
        }
        
        # Disable animations in performance mode
        if performance_mode:
            layout_config['transition'] = {'duration': 0}
            layout_config['animation'] = None
        
        custom_template.layout = layout_config
        
        # Register template
        pio.templates['custom_theme'] = custom_template
        pio.templates.default = 'custom_theme'
    
    def _apply_matplotlib_theme(self):
        """Apply theme to Matplotlib/Seaborn"""
        # Matplotlib rcParams
        plt.rcParams.update({
            'font.family': self.config['font_family'].split(',')[0],
            'font.size': self.config['font_size'],
            'axes.labelsize': self.config['font_size'],
            'axes.titlesize': self.config['title_font_size'],
            'xtick.labelsize': self.config['font_size'] - 1,
            'ytick.labelsize': self.config['font_size'] - 1,
            'legend.fontsize': self.config['font_size'] - 1,
            'figure.titlesize': self.config['title_font_size'],
            'axes.facecolor': self.config['background_color'],
            'figure.facecolor': self.config['background_color'],
            'text.color': self.config['text_color'],
            'axes.labelcolor': self.config['text_color'],
            'xtick.color': self.config['text_color'],
            'ytick.color': self.config['text_color'],
        })
        
        # Seaborn theme
        if self.theme_name == 'dark':
            sns.set_theme(style='darkgrid', palette='muted')
        else:
            sns.set_theme(style='whitegrid', palette='pastel')
    
    def get_plotly_layout(self, title: str = "", **kwargs) -> Dict[str, Any]:
        """
        Get Plotly layout with theme applied
        
        Parameters:
        -----------
        title : str
            Plot title
        **kwargs : dict
            Additional layout parameters
        
        Returns:
        --------
        dict : Layout configuration
        """
        layout = {
            'template': self.config['plotly_template'],
            'font': {
                'family': self.config['font_family'],
                'size': self.config['font_size'],
                'color': self.config['text_color']
            },
            'title': {
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {
                    'size': self.config['title_font_size'],
                    'family': self.config['font_family']
                }
            },
            'hovermode': self.config['hovermode'],
            'hoverlabel': {
                'bgcolor': self.config['hover_bg_color'],
                'font_size': 13,
                'font_family': self.config['font_family']
            },
            'plot_bgcolor': self.config['background_color'],
            'paper_bgcolor': self.config['background_color'],
        }
        
        # Merge with custom kwargs
        layout.update(kwargs)
        
        return layout
    
    def get_color_palette(self, n_colors: int = 10) -> list:
        """
        Get color palette for the current theme
        
        Parameters:
        -----------
        n_colors : int
            Number of colors needed
        
        Returns:
        --------
        list : List of color hex codes
        """
        base_colors = [
            self.config['primary_color'],
            self.config['secondary_color'],
            self.config['accent_color'],
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # Extend if needed
        while len(base_colors) < n_colors:
            base_colors.extend(base_colors)
        
        return base_colors[:n_colors]
    
    def switch_theme(self, new_theme: str, performance_mode: bool = False):
        """
        Switch to a different theme
        
        Parameters:
        -----------
        new_theme : str
            New theme name
        performance_mode : bool
            Enable performance mode
        """
        self._load_theme(new_theme)
        self.theme_name = new_theme
        self._current_theme = new_theme
        self.apply_theme(performance_mode)
        print(f"✨ Switched to {self.config['name']} theme")
    
    @classmethod
    def list_themes(cls):
        """List all available themes"""
        print("🎨 Available Themes:")
        print("=" * 50)
        for theme_key, theme_config in cls.THEMES.items():
            print(f"  • {theme_key:15s} → {theme_config['name']}")
        print("=" * 50)
    
    def export_config(self, filepath: str):
        """
        Export current theme configuration to JSON
        
        Parameters:
        -----------
        filepath : str
            Path to save JSON file
        """
        with open(filepath, 'w') as f:
            json.dump({self.theme_name: self.config}, f, indent=2)
        print(f"✅ Theme config exported to: {filepath}")
    
    @property
    def primary_color(self):
        """Get primary color"""
        return self.config['primary_color']
    
    @property
    def secondary_color(self):
        """Get secondary color"""
        return self.config['secondary_color']
    
    @property
    def accent_color(self):
        """Get accent color"""
        return self.config['accent_color']
    
    @property
    def colorscale(self):
        """Get colorscale name"""
        return self.config['colorscale']


# Global theme instance
_global_theme: Optional[VisualizationTheme] = None


def get_theme() -> VisualizationTheme:
    """Get the global theme instance"""
    global _global_theme
    if _global_theme is None:
        _global_theme = VisualizationTheme(theme="light")
    return _global_theme


def set_theme(theme: str = "light", performance_mode: bool = False, config_path: Optional[str] = None):
    """
    Set global theme
    
    Parameters:
    -----------
    theme : str
        Theme name
    performance_mode : bool
        Enable performance optimizations
    config_path : str, optional
        Path to custom theme config
    """
    global _global_theme
    _global_theme = VisualizationTheme(theme, config_path)
    _global_theme.apply_theme(performance_mode)
    return _global_theme


def switch_theme(new_theme: str, performance_mode: bool = False):
    """
    Switch global theme
    
    Parameters:
    -----------
    new_theme : str
        New theme name
    performance_mode : bool
        Enable performance mode
    """
    theme = get_theme()
    theme.switch_theme(new_theme, performance_mode)


def list_themes():
    """List all available themes"""
    VisualizationTheme.list_themes()

