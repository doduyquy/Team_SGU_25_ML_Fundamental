"""
Test file cho ML Pipeline - House Prices Competition
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('./src'))

from pipeline.pipeline import Pipeline
from model.LightGBM import ModelLightGBM


def load_data():
    """Load data từ raw folder"""
    train_df = pd.read_csv('data/raw/train.csv')
    test_df = pd.read_csv('data/raw/test.csv')
    
    print(f"✅ Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


def create_config():
    """Tạo config cho pipeline"""
    config = {
        "clean": {
            "remove_dup": True,
            "handle_na": True,
            "balance": False,
            "missing_strategy": "house_prices",
            "outlier_method": "house_prices"
        },
        "preprocess": {
            "skip": True,  # Skip preprocessing cho lần đầu
        },
        "feature": {
            "method": None,  # Không dùng feature selection
            "custom_func": None,
        },
        "model": {
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "random_state": 42
            },
            "param_grid": None,  # Tắt grid search để chạy nhanh
            "random_state": 42,
            "test_size": 0.2,
            "model_name": "HousePrice_LightGBM"
        }
    }
    return config


def main():
    """Main function"""
    print("=" * 60)
    print("🏠 HOUSE PRICES ML PIPELINE")
    print("=" * 60)
    
    # 1. Load data
    train_df, test_df = load_data()
    
    # 2. Tạo config
    config = create_config()
    
    # 3. Khởi tạo và chạy pipeline
    print("\n" + "=" * 60)
    print("🚀 Khởi tạo Pipeline...")
    print("=" * 60)
    
    pipeline = Pipeline(
        df=train_df,
        target_col="SalePrice",
        task="regression",
        model_class=ModelLightGBM,
        config=config
    )
    
    # 4. Chạy pipeline
    print("\n" + "=" * 60)
    print("▶️ Chạy Pipeline...")
    print("=" * 60)
    
    model, metrics = pipeline.run()
    
    # 5. In kết quả
    print("\n" + "=" * 60)
    print("📊 KẾT QUẢ")
    print("=" * 60)
    print(f"\nModel: {model}")
    print(f"\nMetrics: {metrics}")
    print("\n✅ Pipeline hoàn tất!")
    print("=" * 60)


if __name__ == "__main__":
    main()
