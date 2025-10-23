from pipeline.BasePipeline import BasePipeline
from model.Random_Forest import ModelRandomForest

class LightGBMPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelRandomForest, experiment_name=experiment_name)
        