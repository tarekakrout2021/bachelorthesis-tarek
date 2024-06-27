import torch.nn as nn

from src.models.VAE import VAE


class Baseline_synthetic(VAE):
    def __init__(self, encoder_layers, decoder_layers, latent_dim):
        super().__init__(
            layer=nn.Linear,
            activation_layer=nn.ReLU(),
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            latent_dim=latent_dim,
        )
        self.latent_dim = latent_dim
