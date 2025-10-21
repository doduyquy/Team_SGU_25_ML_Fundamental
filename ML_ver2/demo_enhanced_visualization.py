"""
🎨 Demo Script: Enhanced Visualization Template
-----------------------------------------------
This script demonstrates all enhanced visualization features.
Run this to see the improvements!

Usage:
    python demo_enhanced_visualization.py
"""

import sys
import os

# Add src to path
sys.path.append('src')

import pandas as pd
import numpy as np

# Import enhanced modules
from utils.display_tools import (
    setup_notebook_style, 
    print_header, 
    print_info,
    show_table,
    show_metrics
)
from core.EDA import EDA
from dashboard import (
    show_target_analysis_dashboard,
    show_correlation_dashboard
)

def main():
    """Main demo function"""
    
    print_header("🎨 ENHANCED VISUALIZATION DEMO", level=1, emoji="🎨")
    print_info("This demo showcases all the enhanced visualization features!", type='info')
    print_info("All plots are interactive - hover, zoom, and explore!", type='success')
    
    # ========================================
    # 1. Load Data
    # ========================================
    print_header("📂 LOADING DATA", level=2, emoji="📂")
    
    try:
        df = pd.read_csv("Data/Raw/train.csv")
        print_info(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns", type='success')
    except FileNotFoundError:
        print_info("❌ Error: Data file not found. Please check the path.", type='error')
        print_info("Expected: Data/Raw/train.csv", type='info')
        return
    
    # ========================================
    # 2. Enhanced Table Display
    # ========================================
    print_header("📋 ENHANCED TABLE DISPLAY", level=2, emoji="📋")
    print_info("Tables now have borders, colors, and better formatting!", type='info')
    
    show_table(
        df.head(10), 
        title="📊 First 10 Rows of Dataset",
        tablefmt='fancy_grid',
        n=10
    )
    
    # ========================================
    # 3. EDA Overview with Interactive Plots
    # ========================================
    print_header("📊 EXPLORATORY DATA ANALYSIS", level=2, emoji="📊")
    
    # Create EDA instance with Plotly
    eda = EDA(df, use_plotly=True)
    
    print_info("Running EDA overview...", type='info')
    eda.overview()
    
    # ========================================
    # 4. Missing Values Analysis
    # ========================================
    print_header("🔍 MISSING VALUES ANALYSIS", level=2, emoji="🔍")
    
    # Create a copy and check missing values
    df_check = df.copy()
    eda_check = EDA(df_check, use_plotly=True)
    eda_check.missing_values()
    
    # ========================================
    # 5. Interactive Correlation Matrix
    # ========================================
    print_header("🔗 INTERACTIVE CORRELATION MATRIX", level=2, emoji="🔗")
    print_info("Hover over cells to see exact correlation values!", type='tip')
    
    # Clean data first
    df_clean = df.select_dtypes(include=[np.number]).dropna()
    eda_clean = EDA(df_clean, use_plotly=True)
    eda_clean.correlation_matrix()
    
    # ========================================
    # 6. Distribution Analysis
    # ========================================
    print_header("📈 DISTRIBUTION ANALYSIS", level=2, emoji="📈")
    print_info("Interactive histogram with box plot margin!", type='info')
    
    if 'SalePrice' in df.columns:
        eda_clean.distribution('SalePrice')
    else:
        print_info("SalePrice column not found, skipping distribution plot", type='warning')
    
    # ========================================
    # 7. Scatter Plot with Trendline
    # ========================================
    print_header("🔍 SCATTER PLOT WITH TRENDLINE", level=2, emoji="🔍")
    
    if 'GrLivArea' in df.columns and 'SalePrice' in df.columns:
        print_info("Showing GrLivArea vs SalePrice with trendline", type='info')
        eda_clean.scatterplot('GrLivArea', 'SalePrice')
    else:
        print_info("Required columns not found, skipping scatter plot", type='warning')
    
    # ========================================
    # 8. Dashboard Views
    # ========================================
    print_header("📊 DASHBOARD VIEWS", level=2, emoji="📊")
    print_info("Comprehensive dashboard with multiple visualizations!", type='info')
    
    # Target analysis dashboard
    if 'SalePrice' in df.columns:
        show_target_analysis_dashboard(df_clean, 'SalePrice')
    
    # Correlation dashboard
    print_info("Showing correlation dashboard with top correlations...", type='info')
    show_correlation_dashboard(df_clean, method='pearson', threshold=0.5)
    
    # ========================================
    # 9. Metrics Display
    # ========================================
    print_header("📈 METRICS DISPLAY", level=2, emoji="📈")
    print_info("Beautiful metrics tables with color gradients!", type='info')
    
    # Example metrics
    example_metrics = {
        'MAE': 25000.5,
        'RMSE': 35000.8,
        'RMSLE': 0.142,
        'R²': 0.851
    }
    
    show_metrics(example_metrics, title="📊 Example Model Metrics")
    
    # ========================================
    # 10. Skewness Analysis
    # ========================================
    print_header("📐 SKEWNESS ANALYSIS", level=2, emoji="📐")
    print_info("Check data distribution skewness", type='info')
    
    eda_clean.check_skewness()
    
    # ========================================
    # Summary
    # ========================================
    print_header("✅ DEMO COMPLETE", level=1, emoji="✅")
    
    print("\n" + "="*80)
    print("🎉 FEATURES DEMONSTRATED:")
    print("="*80)
    print("  ✅ Enhanced table display with borders and colors")
    print("  ✅ Interactive Plotly visualizations (hover, zoom, pan)")
    print("  ✅ Comprehensive correlation analysis")
    print("  ✅ Distribution plots with marginal box plots")
    print("  ✅ Scatter plots with trendlines")
    print("  ✅ Dashboard views with multiple subplots")
    print("  ✅ Beautiful metrics display")
    print("  ✅ Professional styling throughout")
    print("="*80)
    
    print_info("\n💡 TIP: All Plotly plots are interactive!", type='tip')
    print_info("   - Hover over points to see details", type='info')
    print_info("   - Click and drag to zoom", type='info')
    print_info("   - Double-click to reset view", type='info')
    print_info("   - Click camera icon to save as PNG", type='info')
    
    print_info("\n📝 NEXT STEPS:", type='info')
    print("   1. Run the notebook: src/Notebook/EDA.ipynb")
    print("   2. Try model training with enhanced evaluation")
    print("   3. Experiment with SHAP explainability")
    print("   4. Create auto-profiling reports")
    print("   5. Build custom dashboards for your data")
    
    print_info("\n📚 See README.md for complete documentation", type='success')
    print_info("🚀 Happy Machine Learning!\n", type='success')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_info("\n\n⚠️ Demo interrupted by user", type='warning')
    except Exception as e:
        print_info(f"\n\n❌ Error occurred: {str(e)}", type='error')
        print_info("💡 Make sure you have all dependencies installed:", type='info')
        print("   pip install -r requirement.txt")
        raise

