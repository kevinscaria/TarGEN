import abc


class BaseExperiments(object):
    def __init__(self, ):
        self.experiment_name = self.__class__.__name__

    @abc.abstractmethod
    def get_config(self, ):
        raise NotImplementedError

    @abc.abstractmethod
    def generator_function(self,
                           model,
                           step1_prompt=None,
                           step2_prompt=None,
                           step3_prompt=None,
                           step4_prompt=None
                           ):
        raise NotImplementedError
