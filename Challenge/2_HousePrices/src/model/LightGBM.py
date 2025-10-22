import os
import joblib
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
# Import Evaluation Class thực tế từ file src/evaluation.py
from evaluation.Evaluation import Evaluation 

class ModelLightGBM:
    def __init__(self, params=None, param_grid=None, random_state=42, test_size=0.2, 
                 model_name="lightgbm", task="regression"):
        """
        task: 'regression' (Hồi quy) hoặc 'classification' (Phân loại)
        """
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task.lower() # Lưu task hiện tại
        
        self.model = None
        self.best_params = None
        self.metrics = {}

        # Lấy model estimator và scoring dựa trên task
        self.Estimator, self.scoring = self._get_model_and_scoring()

        self.params = params or self._get_default_params()
        self.param_grid = param_grid or self._get_default_param_grid()

    def _get_model_and_scoring(self):
        """Lựa chọn Estimator và Scoring dựa trên task."""
        if self.task == 'classification':
            return lgb.LGBMClassifier, 'roc_auc' # Thường dùng ROC AUC cho phân loại
        elif self.task == 'regression':
            return lgb.LGBMRegressor, 'neg_mean_squared_error' # Thường dùng Neg MSE cho hồi quy
        else:
            raise ValueError("Task phải là 'regression' hoặc 'classification'.")

    def _get_default_params(self):
        """Trả về tham số mặc định cho mô hình."""
        base_params = {"n_estimators": 1000, "learning_rate": 0.05, "random_state": self.random_state}
        if self.task == 'classification':
            base_params['objective'] = 'binary' # Hoặc 'multiclass' tùy vào bài toán
        return base_params

    def _get_default_param_grid(self):
        """Trả về grid params mặc định."""
        return {
            "num_leaves": [31, 50, 70],
            "max_depth": [-1, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 500, 1000]
        }

    def prepare_data(self, df, target_col):
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def tune_params(self, X_train, y_train):
        grid = GridSearchCV(
            # Sử dụng Estimator đã chọn (Classifier hoặc Regressor)
            estimator=self.Estimator(**self.params),
            param_grid=self.param_grid,
            # Sử dụng Scoring đã chọn (neg_mse hoặc roc_auc)
            scoring=self.scoring,
            cv=5,
            n_jobs=-1,
            verbose=1, 
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        return self.best_params

    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        try:
            best_params = self.tune_params(X_train, y_train)
            print("🎯 Best params sau tuning:", best_params)
        except Exception as e:
            print(f"Lỗi tuning: {e}. Sử dụng tham số mặc định.")
            best_params = self.params

        # ===== Loại bỏ random_state nếu có =====
        if "random_state" in best_params:
            print("⚠️ Loại bỏ random_state trùng trong best_params")
            best_params.pop("random_state")

        # Gộp params cuối cùng
        final_params = {**self.params, **best_params}
        final_params["random_state"] = self.random_state

        print("✅ Final params dùng để train:", final_params)

        # ===== Khởi tạo và train model =====
        self.model = self.Estimator(**final_params)
        self.model.fit(X_train, y_train)

        # ===== Đánh giá =====
        self.evaluate(X_test, y_test, feature_names=X_train.columns)

        # ===== Lưu model =====
        # self.save_model()
        return self.model


    def evaluate(self, X_test, y_test, feature_names=None):
        # SỬ DỤNG CLASS EVALUATION THỰC TẾ
        evaluator = Evaluation(self.model, X_test, y_test, 
                               model_name=self.model_name, 
                               task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        print(f"✅ Đã chạy đánh giá đầy đủ cho task: {self.task}.")
        return self.metrics

    def predict(self, X_new):
        X_new = pd.get_dummies(X_new, drop_first=True)
        return self.model.predict(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        
        # Cập nhật: Thêm timestamp vào tên file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Thêm task vào tên file để dễ nhận biết
        path = os.path.join(folder, f"{self.model_name}_{timestamp}.pkl")
        
        joblib.dump(self.model, path)
        print(f"✅ Model saved to: {path}")
        return path
