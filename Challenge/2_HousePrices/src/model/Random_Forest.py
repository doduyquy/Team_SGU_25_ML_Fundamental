import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from evaluation.Evaluation import Evaluation

import optuna # Thêm thư viện Optuna


class ModelRandomForest:
    def __init__(self, model_name="random_forest", task="classification", random_state=42, test_size=0.2):
        self.model_name = model_name
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.best_params = None
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)

        # self.param_grid KHÔNG CẦN THIẾT NỮA
        self.default_params = {"n_estimators": 200, "max_depth": 10}
        self.n_trials = 20 # Số lần thử tối ưu cho Optuna

    def prepare_data(self, df, target_col):
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            X_encoded = pd.get_dummies(X, drop_first=True) 
            X_encoded = X_encoded.fillna(0)
            
            return train_test_split(X_encoded, y, test_size=self.test_size, random_state=self.random_state)

    # ----------------------------------------------------------------------
    # HÀM MỤC TIÊU CHO OPTUNA
    # ----------------------------------------------------------------------
    def _objective_optuna(self, trial, X_train, y_train):
        
        # 1. Định nghĩa Không gian Tìm kiếm Tham số
        param_tune = {
            # Integer
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            
            # Categorical
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            
            # Float
            "max_samples": trial.suggest_float("max_samples", 0.7, 0.95), # Dùng cho Bagging
            
            # Tham số cố định
            "random_state": self.random_state,
            "n_jobs": -1 # Sử dụng tất cả các lõi CPU
        }

        # 2. Lựa chọn Mô hình và Scoring
        if self.task == "classification":
            model = RandomForestClassifier(**param_tune)
            scoring = "roc_auc"
        else:
            model = RandomForestRegressor(**param_tune)
            scoring = "neg_mean_squared_error" 

        # 3. Đánh giá Mô hình bằng Cross-Validation (CV)
        score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=3, 
            scoring=scoring, 
            n_jobs=-1 # Sử dụng n_jobs=-1 ở đây
        ).mean()
        
        # Optuna sẽ tối đa hóa giá trị trả về
        return score

    # ----------------------------------------------------------------------
    # THAY THẾ tune_params BẰNG LOGIC OPTUNA
    # ----------------------------------------------------------------------
    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm tham số tốt nhất cho Random Forest bằng Optuna ({self.n_trials} lần thử)...")
        
        # Hướng tối ưu hóa (maximize cho cả AUC và Neg MSE)
        direction = "maximize" 

        study = optuna.create_study(direction=direction)
        
        # Tối ưu hóa
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train), 
                       n_trials=self.n_trials, 
                       show_progress_bar=True,
                       n_jobs=1) # n_jobs=1 ở đây vì cross_val_score đã dùng n_jobs=-1

        self.best_params = study.best_params
        
        # Thêm các tham số cố định lại vào best_params
        self.best_params["random_state"] = self.random_state
        self.best_params["n_jobs"] = -1
        
        print(f"✅ Best params (Optuna): {self.best_params}")
        return self.best_params

    # ----------------------------------------------------------------------
    # PHẦN train (Giữ nguyên logic nhưng cần sử dụng best_params)
    # ----------------------------------------------------------------------
    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        try:
            best_params = self.tune_params(X_train, y_train)
        except Exception as e:
            print(f"❌ Tuning thất bại: {e}. Sử dụng tham số mặc định.")
            best_params = self.default_params 

        final_params = {k: v for k, v in best_params.items() if k not in ['random_state', 'n_jobs']}

        if self.task == "classification":
            self.model = RandomForestClassifier(**final_params, random_state=self.random_state, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(**final_params, random_state=self.random_state, n_jobs=-1)

        print("🚀 Huấn luyện mô hình Random Forest...")
        self.model.fit(X_train, y_train)
        
        # BƯỚC MỚI: LƯU TRỮ TÊN CỘT ĐÃ ĐƯỢC MÃ HÓA
        self.feature_names_ = X_train.columns.tolist() 

        self.evaluate(X_test, y_test, feature_names=X_train.columns)
        self.save_model()
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
            
        # 1. Xử lý dữ liệu mới (Mã hóa và NaN)
        X_new_encoded = pd.get_dummies(X_new, drop_first=True)
        X_new_encoded = X_new_encoded.fillna(0)
        
        # 2. BƯỚC KHỚP CỘT BẮT BUỘC
        if self.feature_names_ is None:
             raise AttributeError("Chưa có feature_names_. Hãy huấn luyện lại mô hình.")
             
        # Căn chỉnh lại cột: Thêm cột thiếu và loại bỏ cột thừa
        X_new_aligned = X_new_encoded.reindex(columns=self.feature_names_, fill_value=0)

        # 3. Dự đoán
        return self.model.predict(X_new_aligned)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model đã được lưu tại: {path}")
        return path
