import abc


class LossFunctionStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass


class EncoderDecoderStrategy(abc.ABC):
    @abc.abstractmethod
    def encode(self, x):
        pass

    @abc.abstractmethod
    def decode(self, z):
        pass
