from pipeline.BasePipeline import BasePipeline
from model.Naive_Bayes import ModelNaiveBayes

class NBPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelNaiveBayes, experiment_name=experiment_name)
        