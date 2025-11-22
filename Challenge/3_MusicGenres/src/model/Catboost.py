import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from evaluation.Evaluation import Evaluation
import optuna

class ModelCatboost:
    def __init__(self, params=None, random_state=42, test_size=0.2, task="classification", model_name="catboost"):
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model_name = model_name
        self.model = None
        self.best_params= None
        self.metrics = {}
        os.makedirs("experiments/models", exist_ok=True)
        self.categorical_features_indices = None
        
        self.params = params or self._get_default_params()

    def _get_default_params(self):
        return {
            'iterations': 500, 
            'learning_rate': 0.05, 
            'depth': 6,
            'random_state': self.random_state,
            'verbose': False
        }

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 🔹 Tự động nhận dạng các cột phân loại
        self.categorical_features_indices = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        stratify = y if self.task == "classification" else None
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)

    def tune(self, X, y, n_trials=20):
        print(f"🔍 Tuning CatBoost with Optuna ({n_trials} trials)...")
        
        # Detect categorical features
        if self.categorical_features_indices is None:
            self.categorical_features_indices = X.select_dtypes(include=["object", "category"]).columns.tolist()

        def objective(trial):
            param_grid = {
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "iterations": trial.suggest_int("iterations", 500, 2000),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "od_type": "Iter",
                "od_wait": 50,
                "verbose": False,
                "random_seed": self.random_state,
                "bootstrap_type": "Bernoulli"
            }

            if self.task == "classification":
                param_grid["loss_function"] = "MultiClass"
                param_grid["cat_features"] = self.categorical_features_indices
                model = CatBoostClassifier(**param_grid)
                scoring = "f1_macro"
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            else:
                param_grid["loss_function"] = "RMSE"
                param_grid["cat_features"] = self.categorical_features_indices
                model = CatBoostRegressor(**param_grid)
                scoring = "neg_mean_squared_error"
                cv = 3
            
            score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            return score

        direction = "maximize"
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_params['random_state'] = self.random_state
        self.best_params['cat_features'] = self.categorical_features_indices
        self.best_params['bootstrap_type'] = "Bernoulli"
        
        print(f"✅ Best params: {self.best_params}")
        return self.best_params

    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        if X_train is None:
            if df is None or target_col is None:
                raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Detect categorical features if not already detected
        if self.categorical_features_indices is None:
            self.categorical_features_indices = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Determine params
        final_params = self.params.copy()
        if self.best_params:
            final_params.update(self.best_params)
        if params:
            final_params.update(params)
            
        final_params['random_state'] = self.random_state
        final_params['cat_features'] = self.categorical_features_indices
        final_params['verbose'] = 100

        # Fix for CatBoostError: "default bootstrap type (bayesian) doesn't support 'subsample' option"
        if 'subsample' in final_params and 'bootstrap_type' not in final_params:
            final_params['bootstrap_type'] = 'Bernoulli'

        # Initialize model
        if self.task == "classification":
            final_params["loss_function"] = "MultiClass"
            
            # Handle class_weight
            if "class_weight" in final_params:
                cw_option = final_params.pop("class_weight")
                if cw_option == "balanced":
                    from sklearn.utils.class_weight import compute_class_weight
                    import numpy as np
                    classes = np.unique(y_train)
                    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                    class_weights_dict = dict(zip(classes, weights))
                    final_params["class_weights"] = class_weights_dict
                    print("ℹ️ Đã tính toán class_weights cho CatBoost (balanced).")
            
            self.model = CatBoostClassifier(**final_params)
        else:
            final_params["loss_function"] = "RMSE"
            self.model = CatBoostRegressor(**final_params)

        if X_test is not None and y_test is not None:
            self.model.fit(X_train, y_train, eval_set=(X_test, y_test)) 
            self.evaluate(X_test, y_test, feature_names=X_train.columns)
        else:
            self.model.fit(X_train, y_train)
        
        return self.model

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict(X_new)

    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X_new)

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        return path

    @property
    def feature_importances_(self):
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.feature_importances_
