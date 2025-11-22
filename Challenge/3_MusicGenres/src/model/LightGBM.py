import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import optuna
from evaluation.Evaluation import Evaluation 

class ModelLightGBM:
    def __init__(self, params=None, random_state=42, test_size=0.2, 
                 model_name="lightgbm", task="regression"):
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.task = task.lower()
        
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.label_encoders = {}
        
        self.params = params or self._get_default_params()

    def _get_default_params(self):
        """Return default parameters."""
        base_params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1
        }
        if self.task == "classification":
            base_params["objective"] = "multiclass"
            base_params["metric"] = "multi_logloss"
        else:
            base_params["objective"] = "regression"
            base_params["metric"] = "l2"
        return base_params

    def _sanitize_columns(self, X):
        """Clean column names for LightGBM"""
        X = X.copy()
        for char in ['[', ']', '<', '>', ':', '"', '{', '}', ',', ' ']:
            X.columns = X.columns.str.replace(char, '_', regex=False)
        return X

    def _encode_features(self, X, fit=False):
        """Encode categorical features"""
        X = X.copy()
        
        if fit:
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                    else:
                        X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        else:
            for col in X.columns:
                if col in self.label_encoders:
                    try:
                        X[col] = self.label_encoders[col].transform(X[col].astype(str))
                    except ValueError:
                        # Handle unseen labels
                        # For simplicity, assign to -1 (LightGBM handles this)
                        # Ideally, we should fit on full data first or use OrdinalEncoder
                        # Here we map unknown to a new category if possible, or just warn
                         X[col] = X[col].apply(lambda x: self.label_encoders[col].transform([x])[0] if x in self.label_encoders[col].classes_ else -1)

        return self._sanitize_columns(X)

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify = y if self.task == "classification" else None
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)

    def tune(self, X, y, n_trials=20):
        """
        Tune hyperparameters using Optuna.
        """
        print(f"🔍 Tuning LightGBM with Optuna ({n_trials} trials)...")
        
        # Encode X if needed (though tune is usually called inside pipeline which might handle it)
        # But to be safe, let's ensure X is encoded
        X_encoded = self._encode_features(X, fit=True)
        
        def objective(trial):
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 3.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 3.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
                "random_state": self.random_state,
                "n_jobs": -1,
                "verbose": -1
            }
            
            if self.task == "classification":
                param_grid["objective"] = "multiclass"
                param_grid["metric"] = "multi_logloss"
                estimator = lgb.LGBMClassifier(**param_grid)
                scoring = "f1_macro"
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            else:
                param_grid["objective"] = "regression"
                param_grid["metric"] = "l2"
                estimator = lgb.LGBMRegressor(**param_grid)
                scoring = "neg_mean_squared_error"
                cv = 3

            score = cross_val_score(estimator, X_encoded, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            return score

        direction = "maximize" if self.task == "classification" else "maximize" # neg_mse is maximized (closer to 0)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        print(f"✅ Best params: {self.best_params}")
        return self.best_params

    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        """
        Train the model. Can accept df+target_col (will split) or X_train/y_train directly.
        params: dict, if provided, overrides default/tuned params.
        """
        if X_train is None:
            if df is None or target_col is None:
                raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        # Encode features
        X_train = self._encode_features(X_train, fit=True)
        if X_test is not None:
            X_test = self._encode_features(X_test, fit=False)

        # Determine params
        final_params = self.params.copy()
        if self.best_params:
            final_params.update(self.best_params)
        if params:
            final_params.update(params)
            
        final_params["random_state"] = self.random_state
        
        # Initialize model
        # Initialize model
        if self.task == "classification":
            class_weight = final_params.get("class_weight", None)
            if class_weight is not None:
                final_params.pop("class_weight", None)
                self.model = lgb.LGBMClassifier(**final_params, class_weight=class_weight)
            else:
                self.model = lgb.LGBMClassifier(**final_params)
            eval_metric = "multi_logloss"
        else:
            self.model = lgb.LGBMRegressor(**final_params)
            eval_metric = "l2"

        # Train
        if X_test is not None and y_test is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric=eval_metric,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)]
            )
            self.evaluate(X_test, y_test, feature_names=X_train.columns)
        else:
            self.model.fit(X_train, y_train)

        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, 
                               model_name=self.model_name, 
                               task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        X_new = self._encode_features(X_new, fit=False)
        return self.model.predict(X_new)

    def predict_proba(self, X_new):
        X_new = self._encode_features(X_new, fit=False)
        return self.model.predict_proba(X_new)

    @property
    def feature_importances_(self):
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.feature_importances_

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model saved to: {path}")
        return path
