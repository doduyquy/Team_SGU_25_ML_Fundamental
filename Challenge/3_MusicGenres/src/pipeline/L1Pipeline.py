from pipeline.BasePipeline import BasePipeline
from model.L1 import ModelElasticNet

class L1Pipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelElasticNet, experiment_name=experiment_name)