from tracking.Tracking import Tracking
from datetime import datetime
class BasePipeline:
    def __init__(self, model_class, experiment_name="Default_Experiment"):
        self.model_class = model_class
        self.experiment_name = experiment_name

    def run(self, df, target_col, task, model_name, backend="mlflow"):
        run_name = f"{model_name}_{task}_best_{datetime.now().strftime('%H%M%S')}"
        print(f"\n🚀 Start run: {run_name}")

        model = self.model_class(model_name=model_name, task=task)

        # Tracking context
        with Tracking(backend=backend, project=self.experiment_name, run_name=run_name, model=model) as tracker:
            model.train(df, target_col)
            tracker.log_params(model.best_params or model.params)
            tracker.log_metrics(model.metrics)
            tracker.save_model(model)
            tracker.save_data(df.head(100))
