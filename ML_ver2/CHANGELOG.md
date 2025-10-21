# Changelog

All notable changes to this ML Template project will be documented in this file.

## [2.0.0] - Enhanced Visualization Release - 2024

### 🎨 Major Enhancements

#### **Visualization Overhaul**
- ✅ **Plotly Integration**: Replaced all seaborn plots with interactive Plotly visualizations
  - Correlation matrix → Interactive heatmap with hover values
  - Distribution plots → Interactive histograms with marginal box plots
  - Scatter plots → Interactive scatter with trendlines and hover data
  - Box/Violin plots → Interactive box/violin plots with detailed tooltips
  - Feature importance → Interactive bar charts with zoom capability

#### **New Modules**
- ✅ **`src/utils/display_tools.py`**: Helper functions for enhanced display
  - `setup_notebook_style()`: Configure notebook for optimal display
  - `show_table()`: Beautiful table formatting with borders and colors
  - `show_metrics()`: Styled metrics display with color gradients
  - `print_header()`: Formatted section headers with emojis
  - `print_info()`: Colored info messages (success/warning/error/tip)
  - `create_styled_dataframe()`: Styled DataFrames for Jupyter
  - `get_plotly_layout()`: Consistent Plotly layout configuration

- ✅ **`src/dashboard.py`**: Comprehensive dashboard views
  - `quick_eda_dashboard()`: All-in-one EDA dashboard
  - `show_overview_dashboard()`: Distribution overview with subplots
  - `show_correlation_dashboard()`: Correlation heatmap + top correlations
  - `show_target_analysis_dashboard()`: Target variable deep dive
  - `show_model_comparison_dashboard()`: Compare multiple models
  - `show_feature_target_relationship()`: Features vs target analysis

#### **Enhanced EDA Class** (`src/core/EDA.py`)
- ✅ Added `use_plotly` parameter (default: True) with fallback to seaborn
- ✅ **New method**: `run_profiling()` - Auto-generate comprehensive HTML reports with ydata-profiling
- ✅ Enhanced all visualization methods with interactive Plotly plots
- ✅ Improved `overview()` with styled tables and better statistics
- ✅ Enhanced `missing_values()` with percentage display and interactive chart
- ✅ Better error handling and user feedback with colored messages
- ✅ Added limits to prevent memory issues (max_features, top_n parameters)

#### **Enhanced Evaluation Class** (`src/Evaluation/Evaluation.py`)
- ✅ Added `use_plotly` parameter for interactive plots
- ✅ **New method**: `plot_residuals()` - Residuals analysis with dual plots
- ✅ **New method**: `show_shap_summary()` - SHAP explainability analysis
- ✅ Enhanced `plot_regression_fit()` with interactive scatter + perfect prediction line
- ✅ Enhanced `plot_confusion()` with interactive annotated heatmap
- ✅ Enhanced `feature_importance()` with horizontal interactive bars
- ✅ Improved `report()` with styled metrics table
- ✅ Updated `full_evaluation()` with options for residuals and SHAP

#### **Documentation**
- ✅ **README.md**: Comprehensive documentation with examples
- ✅ **QUICK_START.md**: Quick start guide for beginners
- ✅ **CHANGELOG.md**: This file - tracking all changes
- ✅ **demo_enhanced_visualization.py**: Standalone demo script

#### **Enhanced Notebook** (`src/Notebook/EDA.ipynb`)
- ✅ Added setup cell with `setup_notebook_style()`
- ✅ Added demo cells for interactive visualizations
- ✅ Added examples for auto-profiling
- ✅ Added examples for SHAP explainability
- ✅ Added dashboard demonstrations
- ✅ Improved documentation and comments

### 📦 Dependencies Added

```
plotly>=5.18.0
ydata-profiling>=4.6.0
tabulate>=0.9.0
shap>=0.44.0
yellowbrick>=1.5
ipython>=8.12.0
```

### 🔧 Technical Improvements

- ✅ **Backward Compatibility**: All old code still works with `use_plotly=False`
- ✅ **Graceful Fallbacks**: If Plotly not available, falls back to seaborn
- ✅ **Try/Except Import**: Safe imports with informative warnings
- ✅ **Performance**: Added sampling for SHAP to prevent slowdowns
- ✅ **Memory Management**: Limits on features displayed in dashboards
- ✅ **Better Styling**: Consistent theme across all visualizations

### 🎯 Features Summary

| Feature | Old (v1.0) | New (v2.0) |
|---------|-----------|-----------|
| **Plots** | Static (matplotlib/seaborn) | Interactive (Plotly) |
| **Tables** | Plain text | Styled with borders/colors |
| **Profiling** | Manual only | Auto-profiling available |
| **Explainability** | None | SHAP integration |
| **Residuals** | None | Comprehensive analysis |
| **Dashboards** | None | Multiple dashboard views |
| **Styling** | Basic | Professional with themes |
| **User Feedback** | Plain print | Colored messages with emojis |

### 🐛 Bug Fixes

- ✅ Fixed correlation matrix display for large datasets
- ✅ Fixed memory issues with pairplot (now limits to 10 features)
- ✅ Improved error messages for missing columns
- ✅ Better handling of missing values in visualizations

### 🔄 Breaking Changes

**None!** - All changes are backward compatible. Existing code will continue to work.

---

## [1.0.0] - Initial Release

### Features

- ✅ Basic EDA with seaborn visualizations
- ✅ Model training (Lasso, LightGBM)
- ✅ Basic evaluation metrics
- ✅ MLflow tracking integration
- ✅ WandB tracking support
- ✅ GridSearchCV for hyperparameter tuning

### Modules

- `src/core/EDA.py`: Basic EDA functionality
- `src/Evaluation/Evaluation.py`: Basic model evaluation
- `src/model/Lasso.py`: Lasso regression model
- `src/model/LightGBM.py`: LightGBM model
- `src/Tracking/Tracking.py`: Experiment tracking

---

## Future Roadmap

### Planned for v2.1
- [ ] Add more model types (XGBoost, CatBoost, Neural Networks)
- [ ] Add time series analysis module
- [ ] Add automated feature engineering
- [ ] Add model comparison report generator
- [ ] Add hyperparameter tuning with Optuna

### Planned for v3.0
- [ ] Web dashboard with Streamlit/Dash
- [ ] Real-time model monitoring
- [ ] A/B testing framework
- [ ] AutoML integration
- [ ] Cloud deployment templates

---

## Migration Guide (v1.0 → v2.0)

### No changes required for basic usage!

Your existing code will continue to work:

```python
# Old code - still works!
from core.EDA import EDA
eda = EDA(df)
eda.correlation_matrix()  # Will use seaborn by default if plotly not installed
```

### To use new features:

```python
# New code - enhanced visualization
from core.EDA import EDA
eda = EDA(df, use_plotly=True)  # Enable Plotly
eda.correlation_matrix()  # Interactive!
eda.run_profiling()  # New feature!
```

### Recommended Updates:

1. **Install new dependencies**:
   ```bash
   pip install -r requirement.txt
   ```

2. **Add notebook setup** (optional but recommended):
   ```python
   from utils.display_tools import setup_notebook_style
   setup_notebook_style()
   ```

3. **Enable interactive plots**:
   ```python
   eda = EDA(df, use_plotly=True)
   evaluator = Evaluation(..., use_plotly=True)
   ```

---

## Contributors

- Team SGU ML Fundamental

---

## License

MIT License - See LICENSE file for details

