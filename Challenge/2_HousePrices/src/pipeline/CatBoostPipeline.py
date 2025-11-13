from pipeline.BasePipeline import BasePipeline
from model.Catboost import ModelCatBoost

class CatBoostPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelCatBoost, experiment_name=experiment_name)