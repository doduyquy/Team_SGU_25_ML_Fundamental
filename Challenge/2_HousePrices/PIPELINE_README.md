# ML Pipeline - House Prices Competition

## 📋 Tổng quan

Pipeline ML hoàn chỉnh cho bài toán dự đoán giá nhà (House Prices Competition), bao gồm:

- **Data Cleaning**: Xử lý missing values, outliers
- **Preprocessing**: Scaling, normalization
- **Feature Engineering**: Feature selection, custom features
- **Modeling**: LightGBM với hyperparameter tuning
- **Evaluation**: Đánh giá model performance

## 🏗️ Cấu trúc

```
Challenge/2_HousePrices/
├── src/
│   ├── pipeline/
│   │   └── pipeline.py          # Main pipeline class
│   ├── preprocess/
│   │   ├── clean/
│   │   │   └── clean.py         # DataCleaning + Clean class
│   │   ├── preprocess/
│   │   │   └── preprocess.py    # Scaling, normalization
│   │   └── feature/
│   │       └── feature.py       # Feature selection/engineering
│   ├── model/
│   │   └── LightGBM.py          # LightGBM model class
│   └── evaluation/
│       └── evaluation.py        # Evaluation metrics
├── test_pipeline.py             # Demo file
└── PIPELINE_README.md           # File này
```

## 🚀 Cách sử dụng

### 1. Basic usage

```python
import pandas as pd
from src.pipeline.pipeline import Pipeline
from src.model.LightGBM import ModelLightGBM

# Load data
train_df = pd.read_csv('data/raw/train.csv')

# Tạo config
config = {
    "clean": {
        "remove_dup": True,
        "handle_na": True,
        "missing_strategy": "house_prices",
        "outlier_method": "house_prices"
    },
    "preprocess": {
        "skip": False,  # Chạy preprocessing
        "method": "standardize"
    },
    "feature": {
        "method": None  # Không dùng feature selection
    },
    "model": {
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.05
        },
        "random_state": 42
    }
}

# Khởi tạo pipeline
pipeline = Pipeline(
    df=train_df,
    target_col="SalePrice",
    task="regression",
    model_class=ModelLightGBM,
    config=config
)

# Chạy pipeline
model, metrics = pipeline.run()
```

### 2. Run demo file

```bash
cd Challenge/2_HousePrices
python test_pipeline.py
```

### 3. Chạy từng step riêng lẻ

```python
pipeline = Pipeline(df, "SalePrice", config=config)

# Chạy từng step
df_cleaned = pipeline.step_clean()
df_preprocessed = pipeline.step_preprocess()
df_features = pipeline.step_feature()
model, metrics = pipeline.step_train_model()
```

## ⚙️ Configuration options

### Clean step

```python
"clean": {
    "remove_dup": True,                    # Xóa duplicates
    "handle_na": True,                     # Xử lý missing values
    "balance": False,                      # Cân bằng dữ liệu
    "missing_strategy": "house_prices",    # Strategy xử lý missing
    "outlier_method": "house_prices"       # Strategy xử lý outliers
}
```

**Missing strategies:**

- `"house_prices"`: Tối ưu cho House Prices (đã implement)
- `"smart"`: Tự động chọn mean/median/mode
- `"mean"`, `"median"`, `"mode"`: Fill với giá trị cụ thể
- `"drop"`: Drop rows/columns

**Outlier methods:**

- `"house_prices"`: Xử lý theo logic House Prices (remove extreme outliers, log transform)
- `"cap"`: Cap outliers với IQR bounds
- `"remove"`: Remove outliers
- `"transform"`: Log transform

### Preprocess step

```python
"preprocess": {
    "skip": False,              # Skip preprocessing
    "method": "standardize",    # standardize | rescale | normalize
    "params": {}                # Additional params
}
```

### Feature step

```python
"feature": {
    "method": "chi2",           # chi2 | rfe | pca | importance | None
    "params": {"k": 10},        # Params cho method
    "custom_func": None,        # Custom feature function
    "custom_name": None         # Tên custom feature
}
```

### Model step

```python
"model": {
    "params": {                 # LightGBM params
        "n_estimators": 100,
        "learning_rate": 0.05
    },
    "param_grid": None,         # Grid search params (None = no tuning)
    "random_state": 42,
    "test_size": 0.2,
    "model_name": "LightGBM"
}
```

## 📊 Ví dụ đầy đủ

Xem file `test_pipeline.py` để xem ví dụ đầy đủ.

## 🔧 Troubleshooting

### Lỗi import

```python
# Thêm src vào path
import sys
sys.path.append('./src')
```

### Lỗi missing columns

```python
# Check columns trước khi chạy
print(df.columns)
print(target_col in df.columns)
```

### Out of memory

```python
# Giảm số estimators hoặc tắt grid search
"model": {
    "params": {"n_estimators": 50},
    "param_grid": None
}
```

## 📝 Notes

- Pipeline tương thích với House Prices competition
- `Clean` class sử dụng `DataCleaner` internally
- Tất cả classes có method `run()` để consistency
- Config-based approach để dễ customize

## 🎯 Next steps

1. Thêm feature engineering tùy chỉnh
2. Thử các models khác (XGBoost, CatBoost)
3. Tune hyperparameters
4. Ensemble models
