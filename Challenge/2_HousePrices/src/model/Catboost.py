import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from evaluation.Evaluation import Evaluation
import optuna

class ModelCatBoost:
    def __init__(self, random_state=42, test_size=0.2, task="classification", model_name="catboost"):
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.model = None
        self.best_params= None
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)
        self.categorical_features_indices = None
        self.n_trials = 20 # Số lần thử tối ưu cho Optuna
        self.params = {
            'iterations': 500, 
            'learning_rate': 0.05, 
            'depth': 6,
            'random_state': random_state
            # Có thể thêm các tham số cơ bản khác nếu cần
        }

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 🔹 Tự động nhận dạng các cột phân loại
        self.categorical_features_indices = X.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"🔍 Phát hiện {len(self.categorical_features_indices)} cột phân loại:", self.categorical_features_indices)

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
    def _objective_optuna(self, trial, X_train, y_train):
        
        # 1. Định nghĩa Không gian Tìm kiếm Tham số (Chỉ chứa các tham số tuning)
        param_tune = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "iterations": trial.suggest_int("iterations", 500, 2000),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "random_seed": 42,
            "eval_metric": "RMSE",
            "od_type": "Iter",
            "od_wait": 50,
            "verbose": False,
        }

        # 2. Lựa chọn Mô hình và Scoring
        if self.task == "classification":
            model = CatBoostClassifier(
                **param_tune,
                random_seed=42, # Tham số cố định được truyền vào khởi tạo
                eval_metric="Logloss", # Đặt metric cho model khởi tạo
                verbose=False # Tắt verbose trong CV
            )
            scoring = "roc_auc"
        else:
            model = CatBoostRegressor(
                **param_tune,
                random_seed=42,
                eval_metric="RMSE", # Đặt metric cho model khởi tạo
                verbose=False
            )
            scoring = "neg_mean_squared_error"
            
        # 3. Định nghĩa fit_params (Tham số BẮT BUỘC cho fit/cross_val_score)
        fit_params = {
            # BẮT BUỘC: Truyền thông tin cột phân loại tại đây
            'cat_features': self.categorical_features_indices,
            'verbose': False, 
            'od_type': 'Iter',
            'od_wait': 50
        }
        
        # 4. Đánh giá Mô hình bằng Cross-Validation (CV)
        score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=3, 
            scoring=scoring, 
            n_jobs=-1,
            fit_params=fit_params # <-- TRUYỀN cat_features VÀO FIT
        ).mean()
        
        return score

    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm tham số tốt nhất cho CatBoost bằng Optuna ({self.n_trials} lần thử)...")
        
        # Hướng tối ưu hóa (maximize cho cả AUC và Neg MSE)
        direction = "maximize"

        # Tạo Study và tối ưu hóa
        study = optuna.create_study(direction=direction)
        
        # Truyền X_train, y_train vào hàm objective
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train), 
                       n_trials=self.n_trials, 
                       show_progress_bar=True,
                       n_jobs=1) 

        # 4. Lưu kết quả tốt nhất
        self.best_params = study.best_params
        
        # Thêm các tham số cố định vào best_params
        self.best_params['random_state'] = self.random_state
        self.best_params['cat_features'] = self.categorical_features_indices
        self.best_params['verbose'] = 0 # Giữ im lặng trong quá trình train cuối cùng

        print(f"Best params (Optuna): {self.best_params}")
        return self.best_params

    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        try:
            self.best_params = self.tune_params(X_train, y_train)
        except Exception as e:
            print(f" Tuning thất bại: {e}. Sử dụng tham số mặc định.")
            self.best_params = {}
        self.params.update(self.best_params)
        # Tham số cuối cùng
        final_params = {
            **self.best_params, 
            'random_state': self.random_state, 
            'verbose': 100, # Bật verbose cho train cuối cùng để thấy Early Stopping
            'cat_features': self.categorical_features_indices
        }

        # Khởi tạo model cuối cùng
        if self.task == "classification":
            self.model = CatBoostClassifier(**final_params)
        else:
            self.model = CatBoostRegressor(**final_params)

        print(f"Huấn luyện mô hình CatBoost với {len(X_train)} mẫu...")

        self.model.fit(X_train, y_train, eval_set=(X_test, y_test)) 
        
        self.evaluate(X_test, y_test, feature_names=X_train.columns)
        # self.save_model()
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.predict(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        return path
