import os
import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from evaluation.Evaluation import Evaluation
import optuna

class ModelXGBoost:
    def __init__(self, model_name="xgboost", task="classification", random_state=42, test_size=0.2):
        self.model_name = model_name
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.categorical_cols = []
        os.makedirs("experiments/models", exist_ok=True)

        self.default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6
        }
        self.n_trials = 20

    # ----------------------------------------------------------------------
    # Xử lý dữ liệu
    # ----------------------------------------------------------------------
    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Xác định cột phân loại
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Ép kiểu sang category để XGBoost hiểu
        for col in self.categorical_cols:
            X[col] = X[col].astype('category')

        # Lưu danh sách cột train
        self.train_columns = X.columns.tolist()

        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ----------------------------------------------------------------------
    # Hàm mục tiêu cho Optuna
    # ----------------------------------------------------------------------
    def _objective_optuna(self, trial, X_train, y_train):
        param_tune = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": 42,
        }

        if self.task == "classification":
            model = XGBClassifier(**param_tune, use_label_encoder=False, eval_metric="logloss")
            scoring = "roc_auc"
        else:
            model = XGBRegressor(**param_tune, eval_metric="rmse")
            scoring = "neg_mean_squared_error"

        score = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring, n_jobs=-1).mean()
        return score

    # ----------------------------------------------------------------------
    # Tuning bằng Optuna
    # ----------------------------------------------------------------------
    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm tham số tốt nhất cho XGBoost bằng Optuna ({self.n_trials} lần thử)...")

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train),
                       n_trials=self.n_trials, show_progress_bar=True, n_jobs=1)

        self.best_params = study.best_params
        self.best_params["enable_categorical"] = True
        self.best_params["random_state"] = self.random_state

        print(f"✅ Best params (Optuna): {self.best_params}")
        return self.best_params

    # ----------------------------------------------------------------------
    # Train mô hình
    # ----------------------------------------------------------------------
    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        try:
            best_params = self.tune_params(X_train, y_train)
            final_params = {**self.default_params, **best_params}
        except Exception as e:
            print(f"⚠️ Tuning thất bại ({e}). Sử dụng tham số mặc định.")
            final_params = self.default_params

        final_params["random_state"] = self.random_state
        final_params["enable_categorical"] = True
        eval_metric = "logloss" if self.task == "classification" else "rmse"

        if self.task == "classification":
            self.model = XGBClassifier(**final_params, use_label_encoder=False, eval_metric=eval_metric)
        else:
            self.model = XGBRegressor(**final_params, eval_metric=eval_metric)

        print(f"🚀 Huấn luyện mô hình XGBoost với {len(X_train)} mẫu...")

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=200)
        self.evaluate(X_test, y_test, feature_names=X_train.columns)
        return self.model

    # ----------------------------------------------------------------------
    # Tiền xử lý dữ liệu test khi predict
    # ----------------------------------------------------------------------
    def _preprocess(self, X):
        X = X.copy()

        # Nếu cột là category → điền bằng 'Missing'
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        num_cols = X.select_dtypes(include=['number']).columns

        # Điền giá trị thiếu riêng biệt
        for col in cat_cols:
            # Thêm 'Missing' vào danh sách category nếu chưa có
            if X[col].dtype.name == 'category':
                if 'Missing' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories(['Missing'])
            X[col] = X[col].fillna('Missing')

        # Với số → điền bằng 0
        X[num_cols] = X[num_cols].fillna(0)

        # Lưu lại danh sách cột phân loại
        self.categorical_cols = list(cat_cols)

        return X

    # ----------------------------------------------------------------------
    # Dự đoán
    # ----------------------------------------------------------------------
    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")

        X_processed = self._preprocess(X_new)
        return self.model.predict(X_processed)

    # ----------------------------------------------------------------------
    # Đánh giá & Lưu model
    # ----------------------------------------------------------------------
    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model đã được lưu tại: {path}")
        return path
