import torch.nn as nn

from src.models.VAE import VAE


class Baseline_synthetic(VAE):
    def __init__(self):
        super().__init__(
            layer=nn.Linear,
            activation_layer=nn.ReLU(),
            encoder_layers=[200, 200, 200],
            decoder_layers=[200, 200, 200],
        )
