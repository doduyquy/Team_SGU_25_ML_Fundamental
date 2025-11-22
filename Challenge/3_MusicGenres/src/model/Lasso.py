from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
import optuna
import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from evaluation.Evaluation import Evaluation
from typing import Dict, Any, Tuple
import numpy as np

class ModelLasso:
    def __init__(self, 
                 alphas=None, 
                 random_state=42, 
                 test_size=0.2, 
                 model_name="lasso", 
                 task="regression"):
        
        self.alphas = alphas or [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task
        self.n_trials = 30

        self.params = {'alpha': 1.0, 'random_state': random_state, 'max_iter': 5000}
        self.model = None
        self.best_params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.median_values = pd.Series(dtype='float64')

        self.model_dir = "experiments/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # ===== 1. Chuẩn bị dữ liệu =====
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        self.median_values = X.median(numeric_only=True)
        X = X.fillna(self.median_values)
        y = y.fillna(y.mean())

        X = X.select_dtypes(include=['number']).astype('float64')

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ===== 2. Hàm mục tiêu Optuna =====
    def _objective_optuna(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series):
        alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)

        # Dùng pipeline có RobustScaler
        model = make_pipeline(
            RobustScaler(),
            Lasso(alpha=alpha, max_iter=5000, random_state=self.random_state)
        )

        score = cross_val_score(
            model, X_train, y_train, cv=5,
            scoring="neg_mean_squared_error", n_jobs=-1
        ).mean()

        return score

    # ===== 3. Tuning hyperparameter =====
    def tune_params(self, X_train: pd.DataFrame, y_train: pd.Series):
        print(f"🔍 Đang tìm alpha tốt nhất cho Lasso + RobustScaler ({self.n_trials} lần thử)...")

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train),
                       n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)

        self.best_params = study.best_params
        best_alpha = self.best_params["alpha"]
        print(f"✨ Best alpha (Optuna): {best_alpha}")
        return best_alpha

    # ===== 4. Huấn luyện model =====
        # ===== 4. Huấn luyện model (có thể nhận df hoặc X, y) =====
    def train(self, df_or_X, target_col: str = None, y: pd.Series = None):
        """
        df_or_X: có thể là DataFrame (chứa target_col) hoặc X_train
        target_col: tên cột đích (nếu truyền df)
        y: Series hoặc ndarray (nếu đã tách riêng X, y)
        """

        # --- Nhận diện loại dữ liệu đầu vào ---
        if isinstance(df_or_X, pd.DataFrame) and target_col is not None:
            # Trường hợp 1: DataFrame gốc có cột target
            X_train, X_test, y_train, y_test = self.prepare_data(df_or_X, target_col)
        elif isinstance(df_or_X, (pd.DataFrame, np.ndarray)) and y is not None:
            # Trường hợp 2: Đã tách X, y
            X_train, X_test, y_train, y_test = train_test_split(
                df_or_X, y, test_size=self.test_size, random_state=self.random_state
            )
            # Lưu median để dùng cho predict
            if isinstance(X_train, pd.DataFrame):
                self.median_values = X_train.median(numeric_only=True)
            else:
                self.median_values = pd.Series(np.nanmedian(X_train, axis=0))
        else:
            raise ValueError("❌ train() cần truyền (df, target_col) hoặc (X, y).")

        # --- Tuning tham số ---
        best_alpha = self.tune_params(X_train, y_train)

        # --- Huấn luyện mô hình ---
        self.model = make_pipeline(
            RobustScaler(),
            Lasso(alpha=best_alpha, random_state=self.random_state, max_iter=5000)
        )

        try:
            self.model.fit(X_train, y_train)
        except Exception:
            print("⚠️ Lasso không hội tụ, tăng max_iter lên 10000.")
            self.model = make_pipeline(
                RobustScaler(),
                Lasso(alpha=best_alpha, random_state=self.random_state, max_iter=10000)
            )
            self.model.fit(X_train, y_train)

        print("✅ Model Lasso + RobustScaler đã huấn luyện xong.")

        evaluator = Evaluation(self.model, X_test, y_test,
                               model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(
            feature_names=X_train.columns if isinstance(X_train, pd.DataFrame)
            else [f"feature_{i}" for i in range(X_train.shape[1])]
        )

        return self.model


    # ===== 5. Dự đoán =====
    def predict(self, X_new: pd.DataFrame):
        if self.model is None:
            raise ValueError("⚠️ Model chưa được huấn luyện.")
        if self.median_values.empty:
            raise RuntimeError("Median values chưa được lưu. Hãy huấn luyện lại mô hình.")
        
        X_new_processed = X_new.fillna(self.median_values).astype('float64')
        X_new_processed = X_new_processed.reindex(columns=self.model.named_steps['lasso'].feature_names_in_, fill_value=0)
        return self.model.predict(X_new_processed)

    # ===== 6. Lưu model =====
    def save_model(self, filename=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename or f"{self.model_name}_{timestamp}.pkl"
        local_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, local_path)
        print(f"💾 Model đã lưu: {local_path}")
        return local_path
