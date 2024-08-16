from src.models.Bitlinear158 import BitLinear158
from src.models.VAE import VAE
from src.utils.Config import VaeConfig


class BitnetSynthetic(VAE):
    def __init__(self, config: VaeConfig):
        super().__init__(
            config=config,
            layer=BitLinear158,
        )
