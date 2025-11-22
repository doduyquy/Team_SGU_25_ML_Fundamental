import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from evaluation.Evaluation import Evaluation


class ModelNaiveBayes:
    def __init__(self, model_name="naive_bayes", nb_type="gaussian", test_size=0.2, random_state=42, task="classification", params=None):
        """
        nb_type: 'gaussian', 'multinomial', hoặc 'complement'
        """
        self.model_name = model_name
        self.nb_type = nb_type.lower()
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.metrics = {}
        self.best_params = {"type": self.nb_type}
        os.makedirs("experiments/models", exist_ok=True)
        self.params = params

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        if X_train is None:
            if df is None or target_col is None:
                 raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        # Chọn loại NB phù hợp
        if self.nb_type == "gaussian":
            self.model = GaussianNB()
        elif self.nb_type == "multinomial":
            self.model = MultinomialNB()
        elif self.nb_type == "complement":
            self.model = ComplementNB()
        else:
            raise ValueError("nb_type phải là 'gaussian', 'multinomial' hoặc 'complement'.")
        
        # Note: Naive Bayes usually doesn't have many params to tune via set_params, 
        # but we accept them for compatibility.
        if self.params:
            self.model.set_params(**self.params)
        if params:
            self.model.set_params(**params)

        print(f"🚀 Huấn luyện mô hình Naive Bayes ({self.nb_type})...")
        self.model.fit(X_train, y_train)

        # Đánh giá
        if X_test is not None and y_test is not None:
            self.evaluate(X_test, y_test, feature_names=X_train.columns)
            
        self.save_model()
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task="classification")
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.predict(X_new)
    
    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.predict_proba(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_{self.nb_type}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model Naive Bayes đã lưu tại: {path}")
        return path
