# 🎨 Enhanced ML Template: EDA + Model Training

> **Professional Machine Learning Template với Interactive Visualizations**

## ✨ Tính năng nổi bật

### 🔥 **Đã nâng cấp từ bản cũ:**
- ✅ **Interactive Visualizations** - Tất cả biểu đồ giờ đây tương tác được với Plotly
- ✅ **Beautiful Tables** - Bảng có border, màu sắc, dễ đọc với tabulate
- ✅ **Auto Profiling** - Tự động tạo báo cáo EDA toàn diện với ydata-profiling
- ✅ **SHAP Explainability** - Giải thích model với SHAP values và plots
- ✅ **Residuals Analysis** - Phân tích residuals chi tiết cho regression
- ✅ **Dashboard Views** - Tổng hợp nhiều visualizations trong 1 view
- ✅ **Professional Styling** - Giao diện chuyên nghiệp, nhất quán

### 🎯 **Giữ nguyên:**
- ✅ Toàn bộ logic cũ của EDA, Evaluation, Model classes
- ✅ Backward compatibility - Code cũ vẫn chạy được
- ✅ API không thay đổi - Tên hàm và tham số giữ nguyên

---

## 📦 Cài đặt

```bash
# Clone repository
git clone <your-repo>
cd ML_ver2

# Cài đặt dependencies
pip install -r requirement.txt

# Hoặc cài đặt editable mode
pip install -e .
```

---

## 🚀 Quick Start

### 1️⃣ **Setup Notebook Style**
```python
from utils.display_tools import setup_notebook_style

# Configure notebook for best display
setup_notebook_style()
```

### 2️⃣ **Enhanced EDA**
```python
from core.EDA import EDA
import pandas as pd

# Load data
df = pd.read_csv("Data/Raw/train.csv")

# Create EDA instance (use_plotly=True for interactive plots)
eda = EDA(df, use_plotly=True)

# Run analyses
eda.overview()
eda.missing_values()
eda.correlation_matrix()  # Now interactive!
eda.distribution('SalePrice')  # Hover to see values!

# Auto profiling (optional)
eda.run_profiling(output_html="Reports/EDA_Report.html")
```

### 3️⃣ **Enhanced Model Evaluation**
```python
from model.LightGBM import ModelLightGBM
from Evaluation.Evaluation import Evaluation

# Train model
lgbm = ModelLightGBM()
lgbm.train(df, target_col='SalePrice')

# Evaluate with enhanced visualization
evaluator = Evaluation(
    model=lgbm.model,
    X_test=X_test,
    y_test=y_test,
    use_plotly=True  # Interactive plots!
)

# Full evaluation with residuals
metrics = evaluator.full_evaluation(
    feature_names=X_train.columns,
    show_residuals=True,
    show_shap=True  # SHAP explainability
)
```

### 4️⃣ **Dashboard Views**
```python
from dashboard import quick_eda_dashboard, show_model_comparison_dashboard

# Quick comprehensive EDA
quick_eda_dashboard(df, target_col='SalePrice')

# Compare multiple models
models_metrics = {
    'Lasso': {'MAE': 25000, 'RMSE': 35000, 'R²': 0.85},
    'LightGBM': {'MAE': 20000, 'RMSE': 28000, 'R²': 0.91}
}
show_model_comparison_dashboard(models_metrics)
```

---

## 📁 Cấu trúc Project

```
ML_ver2/
├── config/              # Configuration files
│   ├── model.yaml
│   ├── path.yaml
│   └── train.yaml
├── Data/
│   └── Raw/            # Raw datasets
├── src/
│   ├── core/
│   │   └── EDA.py      # ✨ Enhanced with Plotly
│   ├── Evaluation/
│   │   └── Evaluation.py  # ✨ Enhanced with Plotly + SHAP
│   ├── model/
│   │   ├── Lasso.py
│   │   └── LightGBM.py
│   ├── utils/          # 🆕 NEW!
│   │   └── display_tools.py  # Helper functions
│   ├── dashboard.py    # 🆕 Dashboard views
│   ├── Tracking/
│   │   └── Tracking.py  # MLflow/WandB tracking
│   └── Notebook/
│       └── EDA.ipynb    # ✨ Enhanced demo notebook
├── Reports/            # Generated reports
├── mlruns/             # MLflow experiments
├── requirement.txt     # ✨ Updated dependencies
└── README.md          # This file
```

---

## 🎨 Visualization Showcase

### **Before (Seaborn - Static)**
- ❌ Ảnh tĩnh, không tương tác
- ❌ Bảng plain text
- ❌ Font nhỏ, khó đọc

### **After (Plotly - Interactive)**
- ✅ Hover để xem chi tiết
- ✅ Zoom, pan, export
- ✅ Bảng có border, màu sắc
- ✅ Professional styling

---

## 📊 Features Breakdown

### **EDA Class (`src/core/EDA.py`)**

| Method | Description | Visualization |
|--------|-------------|---------------|
| `overview()` | Tổng quan dữ liệu | Styled tables |
| `missing_values()` | Giá trị thiếu | Interactive bar chart |
| `correlation_matrix()` | Ma trận tương quan | Interactive heatmap |
| `distribution(column)` | Phân bố biến | Histogram + box plot |
| `scatterplot(x, y)` | Scatter plot | Interactive scatter with trendline |
| `check_skewness()` | Độ lệch | Interactive bar chart |
| `run_profiling()` | 🆕 Auto profiling | HTML report |

### **Evaluation Class (`src/Evaluation/Evaluation.py`)**

| Method | Description | Visualization |
|--------|-------------|---------------|
| `report()` | Metrics report | Styled table |
| `plot_regression_fit()` | Predicted vs Actual | Interactive scatter + line |
| `plot_confusion()` | Confusion matrix | Interactive heatmap |
| `feature_importance()` | Feature importance | Interactive bar chart |
| `plot_residuals()` | 🆕 Residuals analysis | Dual plots |
| `show_shap_summary()` | 🆕 SHAP explainability | SHAP plots |

### **Dashboard Module (`src/dashboard.py`)** 🆕

| Function | Description |
|----------|-------------|
| `quick_eda_dashboard()` | Comprehensive EDA in one call |
| `show_correlation_dashboard()` | Correlation matrix + top pairs |
| `show_target_analysis_dashboard()` | Target variable analysis |
| `show_model_comparison_dashboard()` | Compare multiple models |
| `show_feature_target_relationship()` | Features vs target relationships |

---

## 🔧 Configuration

### **Sử dụng Plotly (default)**
```python
eda = EDA(df, use_plotly=True)
evaluator = Evaluation(..., use_plotly=True)
```

### **Fallback sang Seaborn**
```python
eda = EDA(df, use_plotly=False)
evaluator = Evaluation(..., use_plotly=False)
```

### **Customize Display**
```python
from utils.display_tools import show_table, print_header

# Custom table display
show_table(df, title="My Data", tablefmt='github')

# Custom headers
print_header("My Section", level=1, emoji="🎯")
```

---

## 📚 Dependencies

**Core:**
- pandas, numpy
- scikit-learn, lightgbm

**Visualization:**
- **plotly** - Interactive plots
- matplotlib, seaborn - Fallback

**Enhanced Features:**
- **ydata-profiling** - Auto profiling
- **tabulate** - Beautiful tables
- **shap** - Model explainability
- **yellowbrick** - ML visualizers

**Tracking:**
- mlflow, wandb

---

## 💡 Tips & Best Practices

### **1. Notebook Setup**
```python
# Always start your notebook with:
from utils.display_tools import setup_notebook_style
setup_notebook_style()
```

### **2. Profiling cho Dataset lớn**
```python
# Sử dụng minimal=True để nhanh hơn
eda.run_profiling(minimal=True)
```

### **3. SHAP cho Sample nhỏ**
```python
# SHAP chậm với dataset lớn, sử dụng sample
X_sample = X_test.sample(n=100, random_state=42)
evaluator.show_shap_summary(X_sample=X_sample)
```

### **4. Export Plots**
```python
# Plotly plots có thể export:
# - Hover trên plot -> Camera icon -> Save as PNG
# - Hoặc programmatically:
fig.write_html("plot.html")
fig.write_image("plot.png")
```

---

## 🎯 Example Workflow

```python
# 1. Setup
from utils.display_tools import setup_notebook_style
setup_notebook_style()

# 2. Load data
import pandas as pd
df = pd.read_csv("Data/Raw/train.csv")

# 3. EDA
from core.EDA import EDA
eda = EDA(df, use_plotly=True)
eda.overview()
eda.correlation_matrix()

# 4. Train model
from model.LightGBM import ModelLightGBM
lgbm = ModelLightGBM()
lgbm.train(df, target_col='SalePrice')

# 5. Evaluate
from Evaluation.Evaluation import Evaluation
X_train, X_test, y_train, y_test = lgbm.prepare_data(df, 'SalePrice')
evaluator = Evaluation(lgbm.model, X_test, y_test, use_plotly=True)
metrics = evaluator.full_evaluation(
    feature_names=X_train.columns,
    show_residuals=True,
    show_shap=True
)

# 6. Track with MLflow
from Tracking.Tracking import Tracking
with Tracking(backend="mlflow", project="MyProject") as tracker:
    tracker.log_params(lgbm.best_params)
    tracker.log_metrics(metrics)
    tracker.save_model(lgbm.model)
```

---

## 🐛 Troubleshooting

### **ImportError: No module named 'plotly'**
```bash
pip install plotly
```

### **SHAP không hoạt động**
```bash
pip install shap
# Lưu ý: SHAP hoạt động tốt nhất với tree-based models
```

### **Tabulate không tìm thấy**
```bash
pip install tabulate
```

### **Memory error khi profiling**
```python
# Sử dụng minimal mode
eda.run_profiling(minimal=True)
```

---

## 📝 Changelog

### **v2.0 - Enhanced Visualization** (Current)
- ✅ Thay thế seaborn bằng plotly cho interactive plots
- ✅ Thêm utils/display_tools.py cho styling
- ✅ Thêm auto profiling với ydata-profiling
- ✅ Thêm SHAP explainability
- ✅ Thêm residuals analysis
- ✅ Thêm dashboard module
- ✅ Enhanced notebook với demos

### **v1.0 - Original Template**
- Basic EDA với seaborn
- Model training với Lasso & LightGBM
- Basic evaluation
- MLflow tracking

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Team SGU ML Fundamental**

---

## 🎉 Enjoy the Enhanced Template!

**Happy Machine Learning! 🚀**

