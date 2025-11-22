from pipeline.BasePipeline import BasePipeline
from model.Gaussian_Naive_Bayes import ModelGaussianNB

class GNBPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelGaussianNB, experiment_name=experiment_name)