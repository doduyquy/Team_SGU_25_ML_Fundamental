import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from evaluation.Evaluation import Evaluation
import optuna

class ModelRandomForest:
    def __init__(self, model_name="random_forest", task="classification", random_state=42, test_size=0.2, params=None):
        self.model_name = model_name
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.best_params = None
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)
        self.params = params
        
        self.default_params = {"n_estimators": 200, "max_depth": 10}
        self.n_trials = 20

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_encoded = pd.get_dummies(X, drop_first=True) 
        X_encoded = X_encoded.fillna(0)
        
        stratify = y if self.task == "classification" else None
        return train_test_split(X_encoded, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)

    def _objective_optuna(self, trial, X_train, y_train):
        param_tune = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            "max_samples": trial.suggest_float("max_samples", 0.7, 0.95),
            "random_state": self.random_state,
            "n_jobs": -1
        }

        if self.task == "classification":
            model = RandomForestClassifier(**param_tune)
            scoring = "f1_macro"
        else:
            model = RandomForestRegressor(**param_tune)
            scoring = "neg_mean_squared_error" 

        score = cross_val_score(
            model, 
            X_train, 
            y_train, 
            cv=3, 
            scoring=scoring, 
            n_jobs=-1
        ).mean()
        
        return score

    def tune_params(self, X_train, y_train):
        print(f"🔍 Đang tìm tham số tốt nhất cho Random Forest bằng Optuna ({self.n_trials} lần thử)...")
        direction = "maximize" 
        study = optuna.create_study(direction=direction)
        study.optimize(lambda trial: self._objective_optuna(trial, X_train, y_train), 
                       n_trials=self.n_trials, 
                       show_progress_bar=True,
                       n_jobs=1)

        self.best_params = study.best_params
        self.best_params["random_state"] = self.random_state
        self.best_params["n_jobs"] = -1
        
        print(f"✅ Best params (Optuna): {self.best_params}")
        return self.best_params

    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        if X_train is None:
            if df is None or target_col is None:
                 raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        final_params = self.default_params.copy()
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
        
        model_params = {k: v for k, v in final_params.items() if k not in ['random_state', 'n_jobs']}

        if self.task == "classification":
            # Handle class_weight
            class_weight = model_params.get("class_weight", None)
            if class_weight is not None:
                # Remove class_weight from model_params to avoid duplication if it's passed as kwarg
                model_params.pop("class_weight", None)
                self.model = RandomForestClassifier(**model_params, class_weight=class_weight, random_state=self.random_state, n_jobs=-1)
            else:
                self.model = RandomForestClassifier(**model_params, random_state=self.random_state, n_jobs=-1)
        else:
            self.model = RandomForestRegressor(**model_params, random_state=self.random_state, n_jobs=-1)

        print("🚀 Huấn luyện mô hình Random Forest...")
        self.model.fit(X_train, y_train)
        
        self.feature_names_ = X_train.columns.tolist() 

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
            
        X_new_encoded = pd.get_dummies(X_new, drop_first=True)
        X_new_encoded = X_new_encoded.fillna(0)
        
        if self.feature_names_ is None:
             raise AttributeError("Chưa có feature_names_. Hãy huấn luyện lại mô hình.")
             
        X_new_aligned = X_new_encoded.reindex(columns=self.feature_names_, fill_value=0)
        return self.model.predict(X_new_aligned)

    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
            
        X_new_encoded = pd.get_dummies(X_new, drop_first=True)
        X_new_encoded = X_new_encoded.fillna(0)
        
        if self.feature_names_ is None:
             raise AttributeError("Chưa có feature_names_. Hãy huấn luyện lại mô hình.")
             
        X_new_aligned = X_new_encoded.reindex(columns=self.feature_names_, fill_value=0)
        return self.model.predict_proba(X_new_aligned)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model đã được lưu tại: {path}")
        return path

    @property
    def feature_importances_(self):
        if self.model is None:
            raise ValueError("Model chưa được huấn luyện.")
        return self.model.feature_importances_
