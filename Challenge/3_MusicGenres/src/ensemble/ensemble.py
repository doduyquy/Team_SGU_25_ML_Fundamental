import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor, StackingClassifier, StackingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from scipy.optimize import minimize
from evaluation.Evaluation import Evaluation

class Ensemble:
    def __init__(self, models, task="auto", method="voting", meta_model=None, test_size=0.2, random_state=42):
        """
        models: list of (name, model) tuples
        task: classification | regression | auto
        method: voting | stacking | bagging | blending
        meta_model: model meta (nếu stacking/blending)
        """
        self.models = models
        self.task = task
        self.method = method
        self.meta_model = meta_model
        self.test_size = test_size
        self.random_state = random_state
        self.ensemble_model = None
        self.metrics = {}

    # === 1. Build ensemble ===
    def build(self):
        if self.method == "voting":
            if self.task == "classification":
                self.ensemble_model = VotingClassifier(estimators=self.models, voting="soft")
            else:
                self.ensemble_model = VotingRegressor(estimators=self.models)

        elif self.method == "stacking":
            if self.task == "classification":
                self.ensemble_model = StackingClassifier(estimators=self.models, final_estimator=self.meta_model)
            else:
                self.ensemble_model = StackingRegressor(estimators=self.models, final_estimator=self.meta_model)

        elif self.method == "bagging":
            base_model = self.models[0][1]
            if self.task == "classification":
                self.ensemble_model = BaggingClassifier(base_model, n_estimators=10, random_state=self.random_state)
            else:
                self.ensemble_model = BaggingRegressor(base_model, n_estimators=10, random_state=self.random_state)
        else:
            raise ValueError("Phương pháp ensemble không hợp lệ.")

        print(f"✅ Đã tạo ensemble theo kiểu {self.method}.")
        return self.ensemble_model

    # === 2. Train + Evaluate ===
    def train_evaluate(self, df, target_col):
        X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        model = self.build()
        model.fit(X_train, y_train)

        evaluator = Evaluation(model, X_test, y_test, model_name="EnsembleModel", task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=X.columns)
        return model, self.metrics

    # === 3. Optimize Weights for Soft Voting ===
    def optimize_weights(self, oof_preds_dict, y_true):
        """
        oof_preds_dict: dict {model_name: oof_probs}
        y_true: labels
        """
        model_names = list(oof_preds_dict.keys())
        oof_list = [oof_preds_dict[k] for k in model_names]
        
        def get_score(weights):
            weights = np.array(weights)
            weights /= weights.sum()
            
            final_prob = np.zeros_like(oof_list[0])
            for i, prob in enumerate(oof_list):
                final_prob += weights[i] * prob
                
            y_pred = np.argmax(final_prob, axis=1)
            return -f1_score(y_true, y_pred, average='macro')

        init_weights = [1/len(model_names)] * len(model_names)
        bounds = [(0, 1)] * len(model_names)
        
        res = minimize(get_score, init_weights, bounds=bounds, method='SLSQP')
        best_weights = res.x / res.x.sum()
        
        return {m: w for m, w in zip(model_names, best_weights)}

    # === 4. Grid Search Weights for Soft Voting ===
    def grid_search_weights(self, oof_preds_dict, y_true, step=0.1):
        """
        oof_preds_dict: dict {model_name: oof_probs}
        y_true: labels
        step: bước nhảy cho trọng số (0.1, 0.05, etc.)
        """
        from itertools import product
        
        model_names = list(oof_preds_dict.keys())
        n_models = len(model_names)
        oof_list = [oof_preds_dict[k] for k in model_names]
        
        # Generate weights
        # Tạo các tổ hợp trọng số có tổng = 1
        # Cách đơn giản: tạo lưới, lọc tổng = 1
        # Để hiệu quả với n_models nhỏ (3-4), ta dùng product
        
        possible_weights = np.arange(0, 1.0 + step/2, step)
        combinations = product(possible_weights, repeat=n_models)
        
        best_score = -1
        best_weights = None
        
        for weights in combinations:
            if not np.isclose(sum(weights), 1.0):
                continue
                
            final_prob = np.zeros_like(oof_list[0])
            for i, prob in enumerate(oof_list):
                final_prob += weights[i] * prob
                
            y_pred = np.argmax(final_prob, axis=1)
            score = f1_score(y_true, y_pred, average='macro')
            
            if score > best_score:
                best_score = score
                best_weights = weights
                
        print(f"✅ Best Grid Search F1: {best_score:.5f}")
        return {m: w for m, w in zip(model_names, best_weights)}
