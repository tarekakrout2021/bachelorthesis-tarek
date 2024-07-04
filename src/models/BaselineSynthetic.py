import torch.nn as nn

from src.models.VAE import VAE
from src.utils.Config import Config


class BaselineSynthetic(VAE):
    def __init__(self, config: Config):
        super().__init__(
            config=config,
            layer=nn.Linear,
        )
