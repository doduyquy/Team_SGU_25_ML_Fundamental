import os
import joblib
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from evaluation.Evaluation import Evaluation


class ModelSVM:
    def __init__(self, model_name="svm", task="classification", random_state=42, test_size=0.2, params=None):
        self.model_name = model_name
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.best_params = None
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)
        self.params = params

        # GridSearch hyperparameters
        if task == "classification":
            self.param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        else:
            self.param_grid = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"],
                "epsilon": [0.1, 0.2, 0.5]
            }

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def tune_params(self, X_train, y_train):
        print("🔍 Đang tìm tham số tốt nhất cho SVM...")

        if self.task == "classification":
            base_model = SVC(random_state=self.random_state, probability=True)
            scoring = "f1_macro" # Changed from roc_auc to f1_macro for multi-class safety
        else:
            base_model = SVR()
            scoring = "neg_mean_squared_error"

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=self.param_grid,
            scoring=scoring,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        print(f"✅ Best params: {self.best_params}")
        return self.best_params

    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        # Support external data for CV
        if X_train is None:
            if df is None or target_col is None:
                 raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        final_params = {}
        if self.params:
            final_params.update(self.params)
        if params:
            final_params.update(params)

        if not params and not self.params:
            try:
                best_params = self.tune_params(X_train, y_train)
                final_params.update(best_params)
            except Exception as e:
                print(f"❌ Tuning thất bại: {e}. Sử dụng tham số mặc định.")
        
        if self.task == "classification":
            self.model = SVC(**final_params, random_state=self.random_state, probability=True)
        else:
            self.model = SVR(**final_params)

        print("🚀 Huấn luyện mô hình SVM...")
        self.model.fit(X_train, y_train)

        if X_test is not None and y_test is not None:
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
        return self.model.predict(X_new)

    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        if self.task != "classification":
            raise ValueError("Regression không hỗ trợ predict_proba.")
        return self.model.predict_proba(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model đã được lưu tại: {path}")
        return path

    @property
    def coef_(self):
        if self.model is None:
             raise ValueError("Model chưa được huấn luyện.")
        if hasattr(self.model, "coef_"):
             return self.model.coef_
        raise AttributeError("Mô hình này không có thuộc tính coef_ (có thể do kernel không phải linear).")
