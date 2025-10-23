import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from evaluation.Evaluation import Evaluation

class ModelLasso:
    def __init__(self, 
                 alphas=None, 
                 random_state=42, 
                 test_size=0.2, 
                 model_name="lasso", 
                 task="regression"):
        """
        Mô hình Lasso Regression tương thích với BasePipeline.
        """
        self.alphas = alphas or [0.001, 0.01, 0.1, 0.5, 1, 5, 10]
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task

        # Thông tin sau huấn luyện
        self.model = None
        self.best_params = {}
        self.metrics = {}

        # Thư mục lưu model
        self.model_dir = "experiments/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # ===== 1. CHUẨN BỊ DỮ LIỆU =====
    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ===== 2. TUNING HYPERPARAMETER =====
    def tune_alpha(self, X_train, y_train):
        grid = GridSearchCV(
            estimator=Lasso(random_state=self.random_state),
            param_grid={"alpha": self.alphas},
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_alpha = grid.best_params_["alpha"]
        self.best_params = {"alpha": best_alpha}
        print(f"🔍 Best alpha: {best_alpha}")
        return best_alpha

    # ===== 3. HUẤN LUYỆN MODEL =====
    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        best_alpha = self.tune_alpha(X_train, y_train)

        self.model = Lasso(alpha=best_alpha, random_state=self.random_state)
        self.model.fit(X_train, y_train)
        print("✅ Model Lasso đã huấn luyện xong.")

        # Đánh giá
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=X_train.columns)
        print("📊 Metrics:", self.metrics)

        return self.model

    # ===== 4. DỰ ĐOÁN =====
    def predict(self, X_new):
        if self.model is None:
            raise ValueError("⚠️ Model chưa được huấn luyện.")
        return self.model.predict(X_new)

    # ===== 5. LƯU MODEL =====
    def save_model(self, filename=None):
        """Hàm này chỉ để pipeline gọi, không tự gọi bên trong."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename or f"{self.model_name}_{timestamp}.pkl"
        local_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, local_path)
        print(f"💾 Model đã lưu: {local_path}")
        return local_path
