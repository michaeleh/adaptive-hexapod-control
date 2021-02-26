from abc import ABC, abstractmethod


class AbstractSimTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def eval(self, *args, **kwargs):
        """
        simulation template process on the input
        """
        raise NotImplementedError('use of abstract method')

    def reset(self):
        pass
