import os
import joblib
import pandas as pd
from datetime import datetime
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from evaluation.Evaluation import Evaluation
import optuna

class ModelElasticNet:
    def __init__(self, 
                 random_state=42, 
                 test_size=0.2, 
                 model_name="elastic_net", 
                 task="regression"):
        """
        Mô hình Elastic Net Regression tối ưu hóa alpha và l1_ratio bằng Optuna.
        """
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task
        self.n_trials = 30 # Số lần thử tối ưu hóa cho Optuna

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
        
        # XỬ LÝ NaN: Elastic Net (và các mô hình tuyến tính khác) KHÔNG xử lý NaN.
        # Phương pháp đơn giản: điền bằng trung vị/trung bình (cần đồng nhất khi dự đoán)
        X = X.fillna(X.median()) 
        y = y.fillna(y.mean())
        
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ===== 2. HÀM MỤC TIÊU CHO OPTUNA =====
    def _objective_optuna(self, trial, X_train, y_train):
        
        # 1. Định nghĩa Không gian Tìm kiếm cho alpha và l1_ratio
        # Alpha (tổng cường độ phạt)
        alpha = trial.suggest_float("alpha", 1e-5, 10.0, log=True)
        # l1_ratio (tỉ lệ phạt L1)
        l1_ratio = trial.suggest_float("l1_ratio", 0.001, 1.0, log=False)

        # 2. Khởi tạo Mô hình ElasticNet
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000, # Tăng số lần lặp để đảm bảo hội tụ
            random_state=self.random_state
        )
        
        # 3. Đánh giá Mô hình bằng Cross-Validation (CV)
        # Sử dụng 'neg_mean_squared_error' vì Optuna tìm cách MAXIMIZE
        score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=5, 
            scoring="neg_mean_squared_error", 
            n_jobs=-1
        ).mean()
        
        # Trả về Negative MSE (Giá trị càng lớn, mô hình càng tốt)
        return score

    # ===== 3. TUNING HYPERPARAMETER BẰNG OPTUNA =====
    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm alpha và l1_ratio tốt nhất cho Elastic Net bằng Optuna ({self.n_trials} lần thử)...")
        
        # Xác định hướng tối ưu hóa (tối đa hóa NEGATIVE MSE)
        direction = "maximize"

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
        self.model = ElasticNet(
            alpha=best_params.get("alpha", 1.0), 
            l1_ratio=best_params.get("l1_ratio", 0.5), # Giá trị mặc định an toàn nếu Optuna lỗi
            random_state=self.random_state, 
            max_iter=5000
        )
        
        try:
            self.model.fit(X_train, y_train)
        except Exception as e:
            print(f"❌ Cảnh báo: Elastic Net không hội tụ. Tăng max_iter.")
            # Thử tăng max_iter và fit lại
            self.model.set_params(max_iter=10000)
            self.model.fit(X_train, y_train)
            
        print("✅ Model Elastic Net đã huấn luyện xong.")

        # Đánh giá
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=X_train.columns)
        print("📊 Metrics:", self.metrics)

        return self.model

    # ===== 5. DỰ ĐOÁN =====
    def predict(self, X_new):
        if self.model is None:
            raise ValueError("⚠️ Model chưa được huấn luyện.")
        
        # Xử lý NaN: Cần đồng nhất cách xử lý NaN với prepare_data (ví dụ: điền bằng trung vị)
        # Lưu ý: Cần đảm bảo X_new có các cột giống hệt X_train và đã được tiền xử lý đầy đủ
        X_new_processed = X_new.fillna(X_new.median()) 
        
        return self.model.predict(X_new_processed)

    # ===== 6. LƯU MODEL =====
    def save_model(self, filename=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = filename or f"{self.model_name}_{timestamp}.pkl"
        local_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, local_path)
        print(f"💾 Model đã lưu: {local_path}")
        return local_path