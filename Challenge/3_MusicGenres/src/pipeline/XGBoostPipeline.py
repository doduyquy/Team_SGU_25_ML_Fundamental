from pipeline.BasePipeline import BasePipeline
from model.XGBoost import ModelXGBoost

class XGBoostPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelXGBoost, experiment_name=experiment_name)
        