from src.models.Bitlinear158 import BitLinear158
from src.models.VAE import VAE


class Bitnet_synthetic(VAE):
    def __init__(self):
        super().__init__(
            layer=BitLinear158,
            encoder_layers=[200, 200, 200],
            decoder_layers=[200, 200, 200],
        )
        print(self)
