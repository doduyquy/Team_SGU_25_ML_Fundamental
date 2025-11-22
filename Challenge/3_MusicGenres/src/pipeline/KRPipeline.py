from pipeline.BasePipeline import BasePipeline
from model.KernelRidge import ModelKernelRidge

class KRPipeline(BasePipeline):
    def __init__(self, experiment_name=None):
        super().__init__(model_class=ModelKernelRidge, experiment_name=experiment_name)