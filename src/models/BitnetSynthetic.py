from src.models.Bitlinear158 import BitLinear158
from src.models.VAE import VAE


class BitnetSynthetic(VAE):
    def __init__(self, encoder_layers, decoder_layers, latent_dim, activation_layer):
        super().__init__(
            layer=BitLinear158,
            activation_layer=activation_layer,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            latent_dim=latent_dim,
        )
        self.latent_dim = latent_dim
        self.activation_layer = activation_layer
