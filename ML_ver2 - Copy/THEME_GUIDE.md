# 🎨 Theme System Guide - ML Template v2.1

## Overview

The enhanced ML Template now includes a comprehensive **Theme System** that provides consistent, beautiful styling across all visualizations, tables, and outputs. Switch between light, dark, and custom themes with a single command!

---

## 🌈 Available Themes

### Built-in Themes:

| Theme | Description | Best For |
|-------|-------------|----------|
| **light** | Modern Light - Clean white background | General use, presentations |
| **dark** | Modern Dark - High contrast dark theme | Night work, reduced eye strain |
| **professional** | Professional - Business-style with serif fonts | Reports, publications |
| **vibrant** | Vibrant - Colorful and energetic | Creative projects, demos |
| **ocean** | Ocean Blue - Calm blue tones | Data science, analytics |
| **sunset** | Sunset Warm - Warm orange/yellow tones | Creative visualization |
| **forest** | Forest Green - Natural green tones | Environmental data |
| **minimal** | Minimal Mono - Minimalist monochrome | Clean, focused analysis |

---

## 🚀 Quick Start

### Setup in Notebook

```python
# Import theme utilities
from utils.display_tools import setup_notebook_style
from utils.theme import list_themes, switch_theme

# Setup with default light theme
setup_notebook_style(theme="light", performance_mode=False)

# View all available themes
list_themes()
```

### Switch Themes

```python
# Switch to dark theme
switch_theme("dark")

# Switch to professional theme with performance mode
switch_theme("professional", performance_mode=True)
```

---

## 📖 Detailed Usage

### 1. Notebook Setup

```python
"""
Setup at the beginning of your notebook
"""
import sys
sys.path.append('..')

from utils.display_tools import setup_notebook_style

# Choose your theme
# Options: 'light', 'dark', 'professional', 'vibrant', 'ocean', 'sunset', 'forest', 'minimal'
setup_notebook_style(theme="light", performance_mode=False)
```

**Parameters:**
- `theme` (str): Theme name
- `performance_mode` (bool): Enable performance optimizations (disable animations, reduce effects)

### 2. View Available Themes

```python
from utils.theme import list_themes

# List all themes with descriptions
list_themes()

# Output:
# 🎨 Available Themes:
# ==================================================
#   • light           → Modern Light
#   • dark            → Modern Dark
#   • professional    → Professional
#   • vibrant         → Vibrant
#   ...
```

### 3. Switch Themes Dynamically

```python
from utils.theme import switch_theme

# Switch to dark theme
switch_theme("dark")

# Switch to vibrant theme with performance mode
switch_theme("vibrant", performance_mode=True)

# After switching, all new visualizations will use the new theme
```

### 4. Use Theme in Code

```python
from utils.theme import get_theme

# Get current theme
theme = get_theme()

# Access theme properties
print(theme.primary_color)      # '#1f77b4'
print(theme.secondary_color)    # '#ff7f0e'
print(theme.colorscale)         # 'Viridis'

# Get color palette
colors = theme.get_color_palette(n_colors=10)

# Get themed layout for Plotly
layout = theme.get_plotly_layout(
    title="My Plot",
    height=600,
    xaxis_title="X Axis",
    yaxis_title="Y Axis"
)

import plotly.graph_objects as go
fig = go.Figure()
fig.update_layout(layout)
```

---

## 🎨 Theme Configuration

### Theme Properties

Each theme includes:

```python
{
    "name": "Modern Light",                              # Display name
    "plotly_template": "plotly_white",                  # Plotly template
    "primary_color": "#1f77b4",                         # Main color
    "secondary_color": "#ff7f0e",                       # Secondary color
    "accent_color": "#2ca02c",                          # Accent color
    "background_color": "#ffffff",                      # Background
    "text_color": "#2c3e50",                           # Text color
    "grid_color": "#e1e4e8",                           # Grid lines
    "font_family": "Inter, Segoe UI, Roboto, ...",     # Font stack
    "font_size": 12,                                    # Base font size
    "title_font_size": 18,                             # Title font size
    "hover_bg_color": "#f8f9fa",                       # Hover background
    "colorscale": "Viridis",                           # Plotly colorscale
    "hovermode": "x unified"                           # Hover mode
}
```

---

## 🛠️ Custom Themes

### Create Custom Theme

Create a JSON file with your theme configuration:

```json
{
  "my_custom_theme": {
    "name": "My Custom Theme",
    "plotly_template": "plotly_white",
    "primary_color": "#FF5733",
    "secondary_color": "#C70039",
    "accent_color": "#900C3F",
    "background_color": "#FFFFFF",
    "text_color": "#2C3E50",
    "grid_color": "#ECF0F1",
    "font_family": "Arial, sans-serif",
    "font_size": 12,
    "title_font_size": 18,
    "hover_bg_color": "#F8F9FA",
    "colorscale": "Hot",
    "hovermode": "closest"
  }
}
```

### Load Custom Theme

```python
from utils.theme import VisualizationTheme

# Load custom theme from file
theme = VisualizationTheme(
    theme="my_custom_theme",
    config_path="path/to/custom_theme.json"
)
theme.apply_theme()
```

### Export Current Theme

```python
from utils.theme import get_theme

theme = get_theme()
theme.export_config("my_exported_theme.json")
```

---

## ⚡ Performance Mode

Performance mode optimizes visualizations for better notebook performance:

### Features:
- ✅ Disables animations
- ✅ Uses `notebook_connected` renderer
- ✅ Reduces transition effects
- ✅ Lighter rendering

### Enable Performance Mode:

```python
# During setup
setup_notebook_style(theme="light", performance_mode=True)

# When switching themes
switch_theme("dark", performance_mode=True)
```

**Recommended for:**
- Large datasets
- Multiple visualizations
- Slower computers
- Remote notebook servers

---

## 📊 Theme Examples

### Example 1: Light Theme (Default)

```python
setup_notebook_style(theme="light")

import plotly.express as px
import pandas as pd

df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()
```

Result: Clean white background, vibrant colors, professional appearance

### Example 2: Dark Theme

```python
switch_theme("dark")

# Same code as above
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
fig.show()
```

Result: Dark background, high contrast colors, easy on eyes

### Example 3: Professional Theme

```python
switch_theme("professional")

# Create dashboard
from dashboard import show_target_analysis_dashboard
show_target_analysis_dashboard(df, 'sepal_length')
```

Result: Business-appropriate styling, serif fonts, conservative colors

### Example 4: Vibrant Theme

```python
switch_theme("vibrant")

# Multiple plots with consistent theming
from core.EDA import EDA
eda = EDA(df, use_plotly=True)
eda.correlation_matrix()
eda.distribution('sepal_length')
```

Result: Energetic colors, rainbow colorscales, creative look

---

## 🎯 Best Practices

### 1. Choose Theme Based on Context

| Context | Recommended Theme |
|---------|------------------|
| Presentations | `light`, `professional` |
| Night work | `dark` |
| Creative demos | `vibrant`, `sunset` |
| Reports | `professional`, `minimal` |
| Data science | `light`, `ocean` |

### 2. Performance Optimization

```python
# For notebooks with many visualizations
setup_notebook_style(theme="light", performance_mode=True)

# Use lazy rendering for multiple plots
from utils.display_tools import batch_show_figures

figures = [fig1, fig2, fig3]
batch_show_figures(figures, delay=0.2, performance_mode=True)
```

### 3. Consistent Styling

```python
# Use theme throughout notebook
from utils.theme import get_theme

theme = get_theme()

# Get colors for custom visualizations
colors = theme.get_color_palette(n_colors=5)

# Use in matplotlib
import matplotlib.pyplot as plt
plt.plot(x, y, color=theme.primary_color)
```

### 4. Accessibility

```python
# For presentations or accessibility needs
switch_theme("professional")  # High contrast, larger fonts
```

---

## 🔧 Advanced Features

### Programmatic Theme Access

```python
from utils.theme import get_theme

theme = get_theme()

# Access all theme properties
config = theme.config

# Get specific colors
primary = theme.primary_color
secondary = theme.secondary_color
accent = theme.accent_color

# Get colorscale
colorscale = theme.colorscale
```

### Apply Theme to Custom Plots

```python
import plotly.graph_objects as go
from utils.theme import get_theme

theme = get_theme()

fig = go.Figure()

# Use themed layout
layout = theme.get_plotly_layout(
    title="Custom Plot",
    height=600,
    xaxis_title="X",
    yaxis_title="Y"
)

fig.update_layout(layout)

# Use themed colors
fig.add_trace(go.Scatter(
    x=x, y=y,
    marker=dict(color=theme.primary_color)
))

fig.show()
```

---

## 📱 Theme Integration with Components

### EDA Class

```python
from core.EDA import EDA

# EDA automatically uses active theme
eda = EDA(df, use_plotly=True)
eda.correlation_matrix()  # Uses current theme colors/layout
```

### Evaluation Class

```python
from Evaluation.Evaluation import Evaluation

# Evaluation uses theme for all plots
evaluator = Evaluation(model, X_test, y_test, use_plotly=True)
evaluator.full_evaluation()  # Themed plots
```

### Dashboard

```python
from dashboard import quick_eda_dashboard

# Dashboard respects active theme
quick_eda_dashboard(df, target_col='target')
```

---

## 🐛 Troubleshooting

### Theme Not Applied

```python
# Make sure to import and call setup
from utils.display_tools import setup_notebook_style
setup_notebook_style(theme="dark")  # Must run this!
```

### Colors Not Changing

```python
# Theme only applies to NEW visualizations
# Rerun cells after switching themes

switch_theme("dark")
# Now rerun visualization cells
```

### Import Errors

```python
# If theme module not found:
import sys
sys.path.append('..')  # Add parent directory to path

from utils.theme import get_theme
```

### Performance Issues

```python
# Enable performance mode
setup_notebook_style(theme="light", performance_mode=True)

# Use minimal renderer
from utils.display_tools import set_plotly_renderer
set_plotly_renderer("notebook_connected")
```

---

## 📝 Summary

### Quick Reference

```python
# Setup
from utils.display_tools import setup_notebook_style
setup_notebook_style(theme="light", performance_mode=False)

# List themes
from utils.theme import list_themes
list_themes()

# Switch theme
from utils.theme import switch_theme
switch_theme("dark")

# Get theme
from utils.theme import get_theme
theme = get_theme()
print(theme.primary_color)

# Performance
setup_notebook_style(theme="light", performance_mode=True)
```

---

## 🎉 Enjoy Beautiful Visualizations!

The theme system makes your notebooks look professional with minimal effort. Experiment with different themes to find what works best for your project!

**Happy Visualizing! 🚀**

---

*ML Template v2.1 | Enhanced Theme System*

