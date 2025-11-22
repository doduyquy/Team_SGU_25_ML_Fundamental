import os
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt # Import Matplotlib cho ví dụ và hỗ trợ kiểu dữ liệu figure
import shutil
# MLflow & WandB
import mlflow
import mlflow.sklearn
import wandb

class Tracking:
    def __init__(self, backend="mlflow", project="ml_tracking", run_name=None, model=None):
        """
        backend: "mlflow" hoặc "wandb"
        project: tên experiment/run
        run_name: tên run cụ thể
        model: mô hình ML (tùy chọn)
        """
        self.backend = backend.lower()
        self.project = project
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model = model
        self.active = None

        # Thư mục local
        self.model_dir = "experiments/models"
        self.data_dir = "experiments/data/processed"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    # ===== CONTEXT MANAGER =====
    def __enter__(self):
        if self.backend == "mlflow":
            mlflow.set_experiment(self.project)
            self.active = mlflow.start_run(run_name=self.run_name)
        elif self.backend == "wandb":
            self.active = wandb.init(project=self.project, name=self.run_name, reinit=True)
        else:
            raise ValueError("Chỉ hỗ trợ backend='mlflow' hoặc 'wandb'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.backend == "mlflow":
            mlflow.end_run()
        elif self.backend == "wandb":
            if self.active:
                self.active.finish()

    # ===== LOG PARAMS =====
    def log_params(self, params: dict):
        # Flatten nested dict nếu có
        flat_params = self._flatten_dict(params)
        if self.backend == "mlflow":
            for k, v in flat_params.items():
                mlflow.log_param(k, v)
        elif self.backend == "wandb":
            wandb.config.update(flat_params)

    # ===== LOG METRICS =====
    def log_metrics(self, metrics: dict):
        flat_metrics = self._flatten_dict(metrics)
        if self.backend == "mlflow":
            for k, v in flat_metrics.items():
                mlflow.log_metric(k, v)
        elif self.backend == "wandb":
            wandb.log(flat_metrics)

    # ===== LOG PLOT (NEW) =====
    def log_plot(self, fig: plt.Figure, plot_name: str):
        """
        Log một biểu đồ (ví dụ: Matplotlib Figure) lên nền tảng tracking.
        fig: Đối tượng Matplotlib Figure.
        plot_name: Tên của biểu đồ/artifact.
        """
        if self.backend == "mlflow":
            # MLflow sử dụng log_figure để lưu figure Matplotlib dưới dạng artifact (thường là .png)
            mlflow.log_figure(fig, artifact_file=f"plots/{plot_name}.png")
            print(f"✅ Đã log biểu đồ '{plot_name}' lên MLflow tại 'plots/{plot_name}.png'.")
        elif self.backend == "wandb":
            # WandB có thể log trực tiếp Matplotlib Figure object
            wandb.log({plot_name: fig})
            print(f"✅ Đã log biểu đồ '{plot_name}' lên WandB.")

    # ===== SAVE MODEL =====
    def save_model(self, model=None, filename=None):
        model = model or self.model
        if model is None:
            print("⚠️ Không có model để lưu.")
            return None

        # 1. Lưu local: Cập nhật để bao gồm cả run_name và timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"{self.run_name}_{timestamp}_model.pkl" # Tên file mới
        filename = filename or default_filename
        local_path = os.path.join(self.model_dir, filename)
        joblib.dump(model, local_path)
        print(f"✅ Mô hình đã lưu local: {local_path}")

        # 2. Log artifact
        if self.backend == "mlflow":
            # log sklearn model chuẩn MLflow
            mlflow.sklearn.log_model(model, artifact_path="model")
        elif self.backend == "wandb":
            # log WandB artifact
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(local_path)
            wandb.log_artifact(artifact)

        return local_path

    # ===== SAVE DATAFRAME =====
    def save_data(self, df: pd.DataFrame, filename=None):
        # Cập nhật để bao gồm cả run_name và timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        default_filename = f"{self.run_name}_{timestamp}_data.csv" # Tên file mới
        filename = filename or default_filename
        file_path = os.path.join(self.data_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"✅ Dữ liệu đã lưu local: {file_path}")

        # Log artifact
        if self.backend == "mlflow":
            mlflow.log_artifact(file_path)
        elif self.backend == "wandb":
            artifact = wandb.Artifact('data', type='dataset')
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
        return file_path

    # ===== LOG ARTIFACT BẤT KỲ =====
    def log_artifact(self, file_path, artifact_name=None, artifact_type="dataset"):
        if not os.path.exists(file_path):
            print(f"⚠️ File {file_path} không tồn tại.")
            return
        if self.backend == "mlflow":
            mlflow.log_artifact(file_path)
        elif self.backend == "wandb":
            artifact_name = artifact_name or os.path.basename(file_path)
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)

    # ====== PRIVATE: FLATTEN DICT =====
    def _flatten_dict(self, d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
