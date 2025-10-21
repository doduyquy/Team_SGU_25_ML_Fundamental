# 📦 Installation Guide

## Quick Installation

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirement.txt
```

### Step 2: Verify Installation

```bash
# Run the demo script
python demo_enhanced_visualization.py
```

If successful, you should see interactive plots and styled tables!

---

## Detailed Installation

### Option 1: Full Installation (Recommended)

Install all features including auto-profiling and SHAP:

```bash
pip install pandas numpy matplotlib seaborn plotly ydata-profiling tabulate shap yellowbrick scikit-learn lightgbm mlflow wandb joblib ipython
```

### Option 2: Minimal Installation

Install only core features without heavy dependencies:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm mlflow joblib
```

Then optionally add:
```bash
# For beautiful tables
pip install tabulate

# For auto-profiling
pip install ydata-profiling

# For model explainability
pip install shap

# For enhanced ML visualizations
pip install yellowbrick
```

### Option 3: Install from requirements.txt

```bash
pip install -r requirement.txt
```

---

## Troubleshooting

### Issue: "No module named 'plotly'"

**Solution:**
```bash
pip install plotly
```

### Issue: "ydata-profiling installation fails"

**Solution:**
Try installing with specific version:
```bash
pip install ydata-profiling==4.6.0
```

Or skip it (profiling will be disabled but other features work):
```bash
# Just don't use eda.run_profiling()
```

### Issue: "SHAP installation fails on Windows"

**Solution:**
1. Install Visual C++ Build Tools
2. Or use conda:
```bash
conda install -c conda-forge shap
```

### Issue: "ImportError with display_tools"

**Solution:**
Make sure you're running from the correct directory:
```bash
cd src
python
>>> from utils.display_tools import setup_notebook_style
```

Or add to path:
```python
import sys
sys.path.append('src')
```

---

## Verification

### Test in Python:

```python
# Test imports
import pandas as pd
import plotly.express as px
from tabulate import tabulate
import shap

print("✅ All core libraries installed successfully!")

# Test enhanced modules
import sys
sys.path.append('src')

from utils.display_tools import setup_notebook_style, print_header
from core.EDA import EDA
from Evaluation.Evaluation import Evaluation

print("✅ All enhanced modules loaded successfully!")
```

### Test in Jupyter:

```python
# Create a new notebook and run:
from utils.display_tools import setup_notebook_style
setup_notebook_style()

import pandas as pd
from core.EDA import EDA

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
eda = EDA(df, use_plotly=True)
eda.overview()
```

If you see styled output, installation is successful! ✅

---

## Environment Setup

### For Jupyter Notebook:

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook
```

### For JupyterLab:

```bash
# Install JupyterLab
pip install jupyterlab

# Start JupyterLab
jupyter lab
```

### For VS Code:

1. Install Python extension
2. Install Jupyter extension
3. Open `.ipynb` files directly in VS Code

---

## Virtual Environment (Recommended)

### Using venv:

```bash
# Create virtual environment
python -m venv ml_env

# Activate (Windows)
ml_env\Scripts\activate

# Activate (Linux/Mac)
source ml_env/bin/activate

# Install dependencies
pip install -r requirement.txt
```

### Using conda:

```bash
# Create conda environment
conda create -n ml_env python=3.9

# Activate
conda activate ml_env

# Install dependencies
pip install -r requirement.txt
```

---

## Post-Installation

### 1. Run Demo Script

```bash
python demo_enhanced_visualization.py
```

This will test all features and show you examples.

### 2. Open Demo Notebook

```bash
cd src/Notebook
jupyter notebook EDA.ipynb
```

Run all cells to see the enhanced features in action.

### 3. Read Documentation

- `README.md` - Full documentation
- `QUICK_START.md` - Quick start guide
- `CHANGELOG.md` - Version history

---

## Updating

To update to the latest version:

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirement.txt --upgrade
```

---

## Uninstallation

To completely remove:

```bash
# Deactivate virtual environment (if using)
deactivate

# Remove virtual environment folder
rm -rf ml_env

# Or remove packages
pip uninstall -r requirement.txt -y
```

---

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Windows, Linux, or macOS
- **RAM**: 4GB minimum (8GB recommended for large datasets)
- **Disk Space**: 500MB for libraries

---

## Optional Tools

### MLflow UI:

```bash
# View experiments in browser
mlflow ui

# Then open http://localhost:5000
```

### Weights & Biases:

```bash
# Login to W&B
wandb login

# Then run experiments with wandb tracking
```

---

## Support

If you encounter any issues:

1. Check `README.md` for documentation
2. Check `QUICK_START.md` for examples
3. Run `demo_enhanced_visualization.py` to verify installation
4. Check error messages for missing libraries
5. Open an issue on GitHub

---

**Happy Machine Learning! 🚀**

