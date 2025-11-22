from pipeline.BasePipeline import BasePipeline
from model.Lasso import ModelLasso

class LassoPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelLasso, experiment_name=experiment_name)