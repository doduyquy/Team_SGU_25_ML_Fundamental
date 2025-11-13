import os
import joblib
import pandas as pd
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
import optuna
# Import Evaluation Class thực tế từ file src/evaluation.py
from evaluation.Evaluation import Evaluation 

class ModelLightGBM:
    def __init__(self, params=None, param_grid=None, random_state=42, test_size=0.2, 
                 model_name="lightgbm", task="regression"):
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task.lower()
        
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.n_trials = 20 # Số lần thử tối ưu cho Optuna

        self.Estimator, self.scoring = self._get_model_and_scoring()

        self.params = params or self._get_default_params()
        # self.param_grid KHÔNG CẦN THIẾT NỮA

    # ... (Các hàm _get_model_and_scoring, _get_default_params giữ nguyên) ...

    def _get_model_and_scoring(self):
        """Lựa chọn Estimator và Scoring dựa trên task."""
        if self.task == 'classification':
            return lgb.LGBMClassifier, 'roc_auc'
        elif self.task == 'regression':
            return lgb.LGBMRegressor, 'neg_mean_squared_error'
        else:
            raise ValueError("Task phải là 'regression' hoặc 'classification'.")

    def _get_default_params(self):
        """Trả về tham số mặc định cho mô hình."""
        base_params = {"n_estimators": 1000, "learning_rate": 0.05, "random_state": self.random_state}
        if self.task == 'classification':
            base_params['objective'] = 'binary'
        return base_params

    def prepare_data(self, df, target_col):
        # LightGBM tự xử lý kiểu 'category', nên chuyển đổi trước khi get_dummies
        # Nhưng vì bạn dùng get_dummies, ta giữ nguyên logic cũ.
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ----------------------------------------------------------------------
    # HÀM MỤC TIÊU CHO OPTUNA
    # ----------------------------------------------------------------------
    def _objective_optuna(self, trial, X_train, y_train):
        
        # 1. Định nghĩa Không gian Tìm kiếm Tham số
        param_tune = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 3.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 3.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "random_state": 42,
        }
        
        # Thêm objective cho Classification
        if self.task == 'classification':
             param_tune['objective'] = 'binary'

        # 2. Khởi tạo Mô hình và Đánh giá bằng CV
        model = self.Estimator(**param_tune)
        
        # score sẽ là roc_auc hoặc neg_mean_squared_error
        score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=3, 
            scoring=self.scoring, 
            n_jobs=-1
        ).mean()
        
        return score

    # ----------------------------------------------------------------------
    # THAY THẾ tune_params BẰNG LOGIC OPTUNA
    # ----------------------------------------------------------------------
    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm tham số tốt nhất cho LightGBM bằng Optuna ({self.n_trials} lần thử)...")

        # Xác định hướng tối ưu hóa (maximize cho cả AUC và Neg MSE)
        direction = "maximize" 

        study = optuna.create_study(direction=direction)
        
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train), 
                       n_trials=self.n_trials, 
                       show_progress_bar=True,
                       n_jobs=1) 

        self.best_params = study.best_params
        
        print(f"✅ Best params (Optuna): {self.best_params}")
        return self.best_params

    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        try:
            best_params = self.tune_params(X_train, y_train)
            print("🎯 Best params sau tuning:", best_params)
        except Exception as e:
            print(f"Lỗi tuning: {e}. Sử dụng tham số mặc định.")
            best_params = {} # Sử dụng dict rỗng nếu tuning thất bại

        # Gộp params cuối cùng (sử dụng best_params)
        final_params = {**self.params, **best_params}
        final_params["random_state"] = self.random_state

        print("✅ Final params dùng để train:", final_params)

        # ===== Khởi tạo và train model =====
        self.model = self.Estimator(**final_params)

        # Cấu hình eval_metric cho Early Stopping
        eval_metric = 'auc' if self.task == 'classification' else 'l2' # l2 = MSE

        # Huấn luyện mô hình với Early Stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)], # Tập xác thực
            eval_metric=eval_metric,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)]
        )

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
