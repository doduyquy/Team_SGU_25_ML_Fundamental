# ⚡ Quick Start Guide

## 🎯 Mục tiêu
Hướng dẫn nhanh sử dụng Enhanced ML Template với interactive visualizations.

---

## 📦 Step 1: Cài đặt

```bash
# Cài đặt tất cả dependencies
pip install -r requirement.txt
```

**Hoặc cài từng thư viện quan trọng:**
```bash
pip install plotly ydata-profiling tabulate shap
```

---

## 🚀 Step 2: Chạy Notebook Demo

```bash
# Navigate to notebook folder
cd src/Notebook

# Mở Jupyter
jupyter notebook EDA.ipynb
```

**Hoặc sử dụng VS Code / JupyterLab**

---

## 💻 Step 3: Code Examples

### **A. EDA với Interactive Plots**

```python
# Setup
from utils.display_tools import setup_notebook_style
setup_notebook_style()

# Load data
import pandas as pd
df = pd.read_csv("../../Data/Raw/train.csv")

# EDA with Plotly
from core.EDA import EDA
eda = EDA(df, use_plotly=True)

# All plots are now interactive!
eda.overview()
eda.correlation_matrix()  # ✨ Hover to see values!
eda.distribution('SalePrice')  # ✨ Zoom in/out!
eda.scatterplot('GrLivArea', 'SalePrice')  # ✨ Interactive!
```

### **B. Auto Profiling Report**

```python
# Generate comprehensive HTML report
eda.run_profiling(
    output_html="../../Reports/EDA_Report.html",
    minimal=True  # Faster mode
)

# Open the HTML file in browser to see interactive report!
```

### **C. Model Training & Enhanced Evaluation**

```python
from model.LightGBM import ModelLightGBM
from Evaluation.Evaluation import Evaluation

# Train
lgbm = ModelLightGBM()
lgbm.train(df, target_col='SalePrice')

# Evaluate with interactive plots
X_train, X_test, y_train, y_test = lgbm.prepare_data(df, 'SalePrice')
evaluator = Evaluation(
    model=lgbm.model,
    X_test=X_test,
    y_test=y_test,
    use_plotly=True  # ✨ Interactive!
)

# Full evaluation
metrics = evaluator.full_evaluation(
    feature_names=X_train.columns,
    show_residuals=True,  # ✨ Residuals analysis
    show_shap=False  # Set True for SHAP (slower)
)
```

### **D. SHAP Explainability** (Optional - may be slow)

```python
# SHAP for model explanation
evaluator.show_shap_summary(
    X_sample=X_test.sample(100, random_state=42),  # Use sample for speed
    plot_type='dot',
    max_display=20
)
```

### **E. Dashboard Views**

```python
from dashboard import (
    quick_eda_dashboard,
    show_target_analysis_dashboard,
    show_correlation_dashboard
)

# Quick comprehensive EDA
quick_eda_dashboard(df, target_col='SalePrice')

# Or specific dashboards
show_target_analysis_dashboard(df, 'SalePrice')
show_correlation_dashboard(df, method='pearson')
```

---

## 📊 Step 4: So sánh Before/After

### **BEFORE (v1.0 - Seaborn)**

```python
# Old way - static plots
eda.correlation_matrix()
# ❌ Static image
# ❌ Can't zoom
# ❌ Can't see exact values easily
```

### **AFTER (v2.0 - Plotly)**

```python
# New way - interactive!
eda.correlation_matrix()
# ✅ Hover to see exact correlation
# ✅ Zoom in/out
# ✅ Export as PNG/HTML
# ✅ Beautiful tooltips
```

---

## 🎨 Step 5: Customization

### **Fallback to Seaborn**
```python
# Nếu không muốn dùng Plotly
eda = EDA(df, use_plotly=False)
evaluator = Evaluation(..., use_plotly=False)
```

### **Custom Table Display**
```python
from utils.display_tools import show_table, print_header, print_info

show_table(df, title="📋 My Data", tablefmt='fancy_grid')
print_header("🎯 MY SECTION", level=1, emoji="🎯")
print_info("Important message!", type='warning')
```

### **Custom Metrics Display**
```python
from utils.display_tools import show_metrics

metrics = {'MAE': 25000, 'RMSE': 35000, 'R²': 0.85}
show_metrics(metrics, title="📈 Model Performance")
```

---

## 🎯 Complete Example Workflow

```python
"""
Complete ML Workflow with Enhanced Visualization
"""

# 1. Setup notebook styling
from utils.display_tools import setup_notebook_style, print_header, print_info
setup_notebook_style()

print_header("🚀 COMPLETE ML WORKFLOW", level=1, emoji="🚀")

# 2. Load data
import pandas as pd
df = pd.read_csv("../../Data/Raw/train.csv")
print_info(f"Loaded {len(df)} rows", type='success')

# 3. EDA
from core.EDA import EDA
print_header("📊 EXPLORATORY DATA ANALYSIS", level=2, emoji="📊")

eda = EDA(df, use_plotly=True)
eda.overview()
eda.missing_values()
eda.correlation_matrix()
eda.distribution('SalePrice')

# Optional: Auto profiling
# eda.run_profiling(output_html="../../Reports/EDA_Report.html", minimal=True)

# 4. Prepare data
df_clean = df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 
                             'FireplaceQu', 'MasVnrType', 'LotFrontage'])

# 5. Train model
from model.LightGBM import ModelLightGBM
print_header("🤖 MODEL TRAINING", level=2, emoji="🤖")

lgbm = ModelLightGBM()
lgbm.train(df_clean, target_col='SalePrice')
print_info("Model training complete!", type='success')

# 6. Evaluation
from Evaluation.Evaluation import Evaluation
print_header("📈 MODEL EVALUATION", level=2, emoji="📈")

X_train, X_test, y_train, y_test = lgbm.prepare_data(df_clean, 'SalePrice')
evaluator = Evaluation(
    model=lgbm.model,
    X_test=X_test,
    y_test=y_test,
    model_name="LightGBM",
    use_plotly=True
)

metrics = evaluator.full_evaluation(
    feature_names=X_train.columns,
    show_residuals=True,
    show_shap=False  # Set True to enable SHAP
)

# 7. Track with MLflow
from Tracking.Tracking import Tracking
print_header("📝 EXPERIMENT TRACKING", level=2, emoji="📝")

with Tracking(backend="mlflow", project="HousePriceProject") as tracker:
    tracker.log_params(lgbm.best_params)
    tracker.log_metrics(metrics)
    tracker.save_model(lgbm.model)

print_info("Experiment logged to MLflow!", type='success')

# 8. Dashboard summary
from dashboard import show_target_analysis_dashboard
print_header("📊 DASHBOARD SUMMARY", level=2, emoji="📊")
show_target_analysis_dashboard(df_clean, 'SalePrice')

print_header("✅ WORKFLOW COMPLETE", level=1, emoji="✅")
print_info("🎉 All visualizations are interactive! Hover, zoom, and explore.", type='success')
```

---

## 📋 Checklist

- [ ] Cài đặt dependencies (`pip install -r requirement.txt`)
- [ ] Chạy notebook demo (`src/Notebook/EDA.ipynb`)
- [ ] Thử interactive plots (hover, zoom)
- [ ] Test auto profiling (optional - tốn thời gian)
- [ ] Train model với enhanced evaluation
- [ ] Xem residuals analysis
- [ ] Test SHAP explainability (optional - tốn thời gian)
- [ ] Xem dashboard views
- [ ] Track experiments với MLflow

---

## 🐛 Common Issues

### **1. ImportError: No module named 'plotly'**
**Fix:**
```bash
pip install plotly
```

### **2. Display tools not working**
**Fix:**
```bash
# Make sure you're in src/ or parent directory
import sys
sys.path.append('..')
from utils.display_tools import setup_notebook_style
```

### **3. SHAP takes too long**
**Fix:**
```python
# Use smaller sample
X_sample = X_test.sample(n=50, random_state=42)
evaluator.show_shap_summary(X_sample=X_sample)
```

### **4. Profiling memory error**
**Fix:**
```python
# Use minimal mode
eda.run_profiling(minimal=True)
```

---

## 💡 Pro Tips

1. **Always start with `setup_notebook_style()`** để có display tốt nhất
2. **Hover over plots** để xem chi tiết data points
3. **Click camera icon** trên Plotly plots để export PNG
4. **Use `show_table()`** thay vì `print(df)` để có bảng đẹp hơn
5. **Run profiling overnight** nếu dataset lớn
6. **SHAP with sample** để tăng tốc độ
7. **Check `mlflow ui`** để xem experiments trong web interface

---

## 🎉 Next Steps

1. Explore the full notebook: `src/Notebook/EDA.ipynb`
2. Read the complete README: `README.md`
3. Customize for your own dataset
4. Build your own dashboards
5. Share your interactive reports!

---

**Happy Machine Learning! 🚀**

**Questions? Check the main README.md or raise an issue!**

