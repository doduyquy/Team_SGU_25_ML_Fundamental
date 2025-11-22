import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from evaluation.Evaluation import Evaluation


class ModelGaussianNB:
    def __init__(self, model_name="gaussian_nb", test_size=0.2, random_state=42, params=None):
        """
        Mô hình Gaussian Naive Bayes cơ bản
        """
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.best_params = {"type": "gaussian"}
        os.makedirs("experiments/models", exist_ok=True)
        self.params = params

    # ====== CHUẨN BỊ DỮ LIỆU ======
    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ====== HUẤN LUYỆN ======
    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        if X_train is None:
            if df is None or target_col is None:
                 raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        print("🚀 Huấn luyện mô hình Gaussian Naive Bayes...")
        self.model = GaussianNB()
        
        if self.params:
            self.model.set_params(**self.params)
        if params:
            self.model.set_params(**params)
            
        self.model.fit(X_train, y_train)

        # Đánh giá
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test, feature_names=X_train.columns)
            
        self.save_model()
        return self.model

    # ====== ĐÁNH GIÁ ======
    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(
            self.model, X_test, y_test,
            model_name=self.model_name, task="classification"
        )
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    # ====== DỰ ĐOÁN ======
    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.predict(X_new)

    # ====== LƯU MÔ HÌNH ======
    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Mô hình GaussianNB đã lưu tại: {path}")
        return path
