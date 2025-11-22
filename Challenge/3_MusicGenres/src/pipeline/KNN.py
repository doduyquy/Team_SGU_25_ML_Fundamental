from pipeline.BasePipeline import BasePipeline
from model.KNN import ModelKNN

class KNNPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelKNN, experiment_name=experiment_name)
        