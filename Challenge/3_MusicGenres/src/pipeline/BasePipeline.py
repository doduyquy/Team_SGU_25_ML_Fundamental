from tracking.Tracking import Tracking
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

class BasePipeline:
    def __init__(self, model_class, experiment_name="Default_Experiment"):
        self.model_class = model_class
        self.experiment_name = experiment_name
        self.model = None
        self.last_run_name = None

    def run(self, df, target_col, task, model_name, backend="mlflow", X_train=None, y_train=None, X_test=None, y_test=None, params=None):
        run_name = f"{model_name}_{task}_best_{datetime.now().strftime('%H%M%S')}"
        print(f"\n🚀 Start run: {run_name}")

        model = self.model_class(model_name=model_name, task=task, params=params)

        # Tracking context
        with Tracking(
            backend=backend,
            project=self.experiment_name,
            run_name=run_name,
            model=model
        ) as tracker:
            model.train(df, target_col, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, params=params)
            tracker.log_params(model.best_params or model.params)
            tracker.log_metrics(model.metrics)
            tracker.save_model(model)
            if df is not None:
                tracker.save_data(df.head(100))
            elif X_train is not None and hasattr(X_train, "head"):
                tracker.save_data(X_train.head(100))

        # Lưu lại model nếu cần dùng tiếp trong notebook
        self.model = model
        self.last_run_name = run_name
        return model

    def run_cv(self, df, target_col, task, model_name, cv=5, backend="mlflow", test_df=None, model_params=None):
        """
        Run Cross-Validation and return OOF predictions + Test predictions (if test_df provided)
        """
        run_name = f"{model_name}_{task}_CV_{datetime.now().strftime('%H%M%S')}"
        print(f"\n🚀 Start CV run: {run_name} (CV={cv})")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Setup CV
        if task == "classification":
            kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            
        # We need to know n_classes beforehand for array initialization
        classes = np.unique(y)
        n_classes = len(classes)
        
        if task == "classification":
            oof_probs = np.zeros((len(df), n_classes))
            if test_df is not None:
                test_probs = np.zeros((len(test_df), n_classes))
        else:
            oof_probs = np.zeros(len(df))
            if test_df is not None:
                test_preds = np.zeros(len(test_df))
            
        metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\n🔄 Fold {fold+1}/{cv}")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize model for this fold with params
            model = self.model_class(model_name=f"{model_name}_fold{fold+1}", task=task, params=model_params)
            
            # Train
            model.train(df=None, target_col=None, X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, params=model_params)
            
            # Predict OOF
            if task == "classification":
                try:
                    probs = model.predict_proba(X_val)
                    oof_probs[val_idx] = probs
                    
                    if test_df is not None:
                        test_probs += model.predict_proba(test_df) / cv
                except:
                    print("⚠️ Model does not support predict_proba, using predict (one-hot)")
                    pass
            else:
                preds = model.predict(X_val)
                oof_probs[val_idx] = preds
                
                if test_df is not None:
                    test_preds += model.predict(test_df) / cv
                
            metrics_list.append(model.metrics)
            
        # Log average metrics
        avg_metrics = pd.DataFrame(metrics_list).mean().to_dict()
        print(f"\n📊 Average CV Metrics: {avg_metrics}")
        
        if test_df is not None:
            if task == "classification":
                return oof_probs, test_probs, avg_metrics
            else:
                return oof_probs, test_preds, avg_metrics
        
        return oof_probs, avg_metrics
