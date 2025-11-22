import os
import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from evaluation.Evaluation import Evaluation
import optuna

class ModelXGBoost:
    def __init__(self, params=None, model_name="xgboost", task="classification", random_state=42, test_size=0.2):
        self.model_name = model_name
        self.task = task
        self.random_state = random_state
        self.test_size = test_size
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.categorical_cols = []
        os.makedirs("experiments/models", exist_ok=True)

        self.params = params or self._get_default_params()

    def _get_default_params(self):
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": self.random_state,
            "n_jobs": -1,
            "enable_categorical": True
        }

    def _ensure_categorical(self, X):
        X = X.copy()
        # Auto-detect object columns if not set
        if not self.categorical_cols:
            self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
        return X

    def prepare_data(self, df, target_col):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Xác định cột phân loại
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Ép kiểu sang category để XGBoost hiểu
        X = self._ensure_categorical(X)

        stratify = y if self.task == "classification" else None
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)

    def tune(self, X, y, n_trials=20):
        print(f"🔍 Tuning XGBoost with Optuna ({n_trials} trials)...")
        
        X = self._ensure_categorical(X)

        def objective(trial):
            param_grid = {
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
                "random_state": self.random_state,
                "n_jobs": -1
            }

            if self.task == "classification":
                # Multi-class objective
                param_grid["objective"] = "multi:softprob"
                param_grid["eval_metric"] = "mlogloss"
                model = XGBClassifier(**param_grid, use_label_encoder=False)
                scoring = "f1_macro"
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
            else:
                param_grid["objective"] = "reg:squarederror"
                param_grid["eval_metric"] = "rmse"
                model = XGBRegressor(**param_grid)
                scoring = "neg_mean_squared_error"
                cv = 3

            score = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
            return score

        direction = "maximize" # f1_macro or neg_mse (both maximize)
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.best_params = study.best_params
        self.best_params["enable_categorical"] = True
        self.best_params["random_state"] = self.random_state
        
        print(f"✅ Best params: {self.best_params}")
        return self.best_params

    def train(self, df=None, target_col=None, X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        if X_train is None:
            if df is None or target_col is None:
                raise ValueError("Must provide either (df, target_col) or (X_train, y_train)")
            X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)

        # Ensure categorical types
        X_train = self._ensure_categorical(X_train)
        if X_test is not None:
            X_test = self._ensure_categorical(X_test)

        # Determine params
        final_params = self.params.copy()
        if self.best_params:
            final_params.update(self.best_params)
        if params:
            final_params.update(params)

        final_params["random_state"] = self.random_state
        final_params["enable_categorical"] = True
        
        # Handle class_weight for XGBoost (via sample_weight)
        sample_weight = None
        if self.task == "classification" and "class_weight" in final_params:
            cw_option = final_params.pop("class_weight")
            if cw_option == "balanced":
                from sklearn.utils.class_weight import compute_sample_weight
                # Compute sample weights based on y_train
                # Note: y_train might be pandas Series or numpy array
                sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
                print("ℹ️ Đã tính toán sample_weight cho XGBoost (balanced).")

        if self.task == "classification":
            final_params["objective"] = "multi:softprob"
            final_params["eval_metric"] = "mlogloss"
            self.model = XGBClassifier(**final_params, use_label_encoder=False)
        else:
            final_params["objective"] = "reg:squarederror"
            final_params["eval_metric"] = "rmse"
            self.model = XGBRegressor(**final_params)

        if X_test is not None and y_test is not None:
            self.model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_test, y_test)], verbose=False)
            self.evaluate(X_test, y_test, feature_names=X_train.columns)
        else:
            self.model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
            
        return self.model

    def _preprocess(self, X):
        X = X.copy()
        X = self._ensure_categorical(X)
        return X

    def predict(self, X_new):
        if self.model is None:
            raise ValueError("Model not trained.")
        X_processed = self._preprocess(X_new)
        return self.model.predict(X_processed)

    def predict_proba(self, X_new):
        if self.model is None:
            raise ValueError("Model not trained.")
        X_processed = self._preprocess(X_new)
        return self.model.predict_proba(X_processed)

    def evaluate(self, X_test, y_test, feature_names=None):
        evaluator = Evaluation(self.model, X_test, y_test, model_name=self.model_name, task=self.task)
        self.metrics = evaluator.full_evaluation(feature_names=feature_names)
        return self.metrics

    def save_model(self, folder="experiments/models"):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.model_name}_best.pkl")
        joblib.dump(self.model, path)
        print(f"💾 Model saved to: {path}")
        return path

    @property
    def feature_importances_(self):
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.feature_importances_
