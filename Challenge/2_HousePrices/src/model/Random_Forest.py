import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from evaluation.Evaluation import Evaluation


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

        # Lưới tham số cho GridSearch
        self.param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def tune_params(self, X_train, y_train):
        print("🔍 Đang tìm tham số tốt nhất cho Random Forest...")

        if self.task == "classification":
            base_model = RandomForestClassifier(random_state=self.random_state)
            scoring = "roc_auc"
        else:
            base_model = RandomForestRegressor(random_state=self.random_state)
            scoring = "neg_mean_squared_error"

        grid = GridSearchCV(
            base_model,
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

    def train(self, df, target_col):
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        best_params = self.tune_params(X_train, y_train)

        if self.task == "classification":
            self.model = RandomForestClassifier(**best_params, random_state=self.random_state)
        else:
            self.model = RandomForestRegressor(**best_params, random_state=self.random_state)

        print("🚀 Huấn luyện mô hình Random Forest...")
        self.model.fit(X_train, y_train)

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

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model đã được lưu tại: {path}")
        return path
