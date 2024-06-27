import torch
import torch.nn as nn

from src.models.Bitlinear158 import BitLinear158
from src.models.VAE import VAE


class Bitnet_mnist(VAE):
    def __init__(self, encoder_layers, decoder_layers, latent_dim, activation_layer):
        super().__init__(
            layer=BitLinear158,
            activation_layer=activation_layer,
            input_dim=784,
            latent_dim=latent_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
        )
        self.activation_layer = activation_layer
        self.latent_dim = latent_dim
        self.decoder.append(nn.Sigmoid())

    def sample(self, n_samples=100, device="cpu"):
        """
        Sample from p(z) and decode.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(
                device
            )  # Sample from N(0, I)
            self.to(device)
            sampled_data = self.decode(z).cpu()

        sampled_data = sampled_data.view(n_samples, 28, 28)

        return sampled_data
