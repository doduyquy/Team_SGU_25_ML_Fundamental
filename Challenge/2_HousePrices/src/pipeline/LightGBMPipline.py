from pipeline.BasePipeline import BasePipeline
from model.LightGBM import ModelLightGBM

class LightGBMPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelLightGBM, experiment_name=experiment_name)
        