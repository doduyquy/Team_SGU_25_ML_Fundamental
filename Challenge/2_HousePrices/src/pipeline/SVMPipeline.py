from pipeline.BasePipeline import BasePipeline
from model.SVM import ModelSVM

class SVMPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelSVM, experiment_name=experiment_name)
        