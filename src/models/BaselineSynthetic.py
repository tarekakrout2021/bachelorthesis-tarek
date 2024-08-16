import torch.nn as nn

from src.models.VAE import VAE
from src.utils.Config import VaeConfig


class BaselineSynthetic(VAE):
    def __init__(self, config: VaeConfig):
        super().__init__(
            config=config,
            layer=nn.Linear,
        )
