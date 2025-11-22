import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from evaluation.Evaluation import Evaluation


class ModelKNN:
    def __init__(self, model_name="knn", test_size=0.2, random_state=42, n_neighbors_list=None, task="classification", params=None):
        """
        Mô hình K-Nearest Neighbors (KNN) cơ bản, hỗ trợ GridSearch tìm số láng giềng tối ưu.
        """
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.n_neighbors_list = n_neighbors_list or list(range(1, 21))
        self.model = None
        self.best_params = {}
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)
        self.params = params

    # ====== CHUẨN BỊ DỮ LIỆU ======
    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    # ====== TÌM THAM SỐ TỐI ƯU ======
    def tune_hyperparams(self, X_train, y_train):
        print("🔍 Đang tìm số láng giềng tối ưu (k)...")
        grid = GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid={"n_neighbors": self.n_neighbors_list},
            scoring="f1_macro", # Changed to f1_macro
            cv=5,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        self.best_params = grid.best_params_
        print(f"✅ K tốt nhất: {self.best_params['n_neighbors']}")
        return self.best_params

    # ====== HUẤN LUYỆN ======
    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
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
            best_params = self.tune_hyperparams(X_train, y_train)
            final_params.update(best_params)
        
        self.model = KNeighborsClassifier(**final_params)
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
    
    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.predict_proba(X_new)

    # ====== LƯU MÔ HÌNH ======
    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Mô hình KNN đã lưu tại: {path}")
        return path
