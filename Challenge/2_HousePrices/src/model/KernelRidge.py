import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.kernel_ridge import KernelRidge # Import mô hình KRR
from sklearn.model_selection import train_test_split, cross_val_score
from evaluation.Evaluation import Evaluation
import optuna
from sklearn.preprocessing import StandardScaler # Thường cần scale dữ liệu cho KRR

class ModelKernelRidge:
    def __init__(self, 
                 random_state=42, 
                 test_size=0.2, 
                 model_name="kernel_ridge", 
                 task="regression",
                 kernel="rbf"): # Thường dùng RBF kernel
        """
        Mô hình Kernel Ridge Regression tối ưu hóa alpha và gamma (nếu dùng RBF).
        """
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task
        self.kernel = kernel
        self.n_trials = 30 # Số lần thử tối ưu hóa cho Optuna
        
        # Thêm scaler để tiền xử lý dữ liệu số (cần thiết cho KRR)
        self.scaler = StandardScaler() 

        # Thông tin sau huấn luyện
        self.model = None
        self.best_params = {}
        self.metrics = {}

        # Thư mục lưu model
        self.model_dir = "experiments/models"
        os.makedirs(self.model_dir, exist_ok=True)

    # ===== 1. CHUẨN BỊ DỮ LIỆU =====
    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()
        
        # XỬ LÝ NaN: KRR không xử lý NaN.
        X = X.fillna(X.median()) 
        y = y.fillna(y.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # SCALING DỮ LIỆU (RẤT QUAN TRỌNG CHO KRR)
        numeric_cols = X_train.select_dtypes(include=['number']).columns
        
        # Fit scaler chỉ trên tập train
        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        # Transform tập test
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        return X_train, X_test, y_train, y_test

    # ===== 2. HÀM MỤC TIÊU CHO OPTUNA =====
    def _objective_optuna(self, trial, X_train, y_train):
        
        # 1. Định nghĩa Không gian Tìm kiếm 
        # Alpha (Tham số điều chuẩn)
        alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
        
        param_tune = {"alpha": alpha, "kernel": self.kernel}
        
        # Gamma (Tham số kernel RBF) chỉ được điều chỉnh nếu dùng kernel RBF
        if self.kernel == "rbf":
            gamma = trial.suggest_float("gamma", 1e-3, 1.0, log=True)
            param_tune["gamma"] = gamma

        # 2. Khởi tạo Mô hình KernelRidge
        model = KernelRidge(**param_tune)
        
        # 3. Đánh giá Mô hình bằng Cross-Validation (CV)
        score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=5, 
            scoring="neg_mean_squared_error", 
            n_jobs=-1
        ).mean()
        
        return score

    # ===== 3. TUNING HYPERPARAMETER BẰNG OPTUNA =====
    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm alpha và gamma tốt nhất cho KRR ({self.kernel}) bằng Optuna ({self.n_trials} lần thử)...")
        
        direction = "maximize" # Tối đa hóa NEGATIVE MSE

        study = optuna.create_study(direction=direction)
        
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train), 
                       n_trials=self.n_trials, 
                       show_progress_bar=True,
                       n_jobs=1) 

        self.best_params = study.best_params
        
        print(f"✨ Best params (Optuna): {self.best_params}")
        return self.best_params

    # ===== 4. HUẤN LUYỆN MODEL =====
    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        best_params = self.tune_params(X_train, y_train)

        # Khởi tạo và Huấn luyện Model cuối cùng
        # Thêm kernel vào best_params để khởi tạo
        final_params = {**best_params, "kernel": self.kernel}
        
        self.model = KernelRidge(**final_params)
        
        self.model.fit(X_train, y_train)
            
        print("✅ Model Kernel Ridge đã huấn luyện xong.")

        # Đánh giá
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=X_train.columns)
        print("📊 Metrics:", self.metrics)

        return self.model

    # ===== 5. DỰ ĐOÁN =====
    def predict(self, X_new):
        if self.model is None:
            raise ValueError("⚠️ Model chưa được huấn luyện.")
        
        # XỬ LÝ DỮ LIỆU MỚI: BẮT BUỘC PHẢI TIỀN XỬ LÝ (fillna và scaling)
        X_new_processed = X_new.copy()
        
        # 1. Fillna (Đồng nhất với prepare_data)
        X_new_processed = X_new_processed.fillna(X_new_processed.median()) 
        
        # 2. SCALING (Sử dụng self.scaler đã fit trên tập train)
        numeric_cols = X_new_processed.select_dtypes(include=['number']).columns
        X_new_processed[numeric_cols] = self.scaler.transform(X_new_processed[numeric_cols])
        
        return self.model.predict(X_new_processed)

    # ===== 6. LƯU MODEL =====
    def save_model(self, filename=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename or f"{self.model_name}_{timestamp}.pkl"
        local_path = os.path.join(self.model_dir, filename)
        
        # Lưu cả model và scaler
        joblib.dump({"model": self.model, "scaler": self.scaler}, local_path) 
        print(f"💾 Model và Scaler đã lưu: {local_path}")
        return local_path