import torch
import torch.nn as nn

from src.models.Bitlinear158 import BitLinear158
from src.models.VAE import VAE


class Bitnet_mnist(VAE):
    def __init__(self):
        super().__init__(
            layer=BitLinear158,
            input_dim=784,
            latent_dim=2,
            encoder_layers=[400, 400, 200],
            decoder_layers=[200, 400, 400, 400],
        )
        self.decoder.append(nn.Sigmoid())
        print(self)

    def sample(self, n_samples=100, device="cpu"):
        """
        Sample from p(z) and decode.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, 2).to(device)  # Sample from N(0, I)
            self.to(device)
            sampled_data = self.decode(z).cpu()

        sampled_data = sampled_data.view(n_samples, 28, 28)

        return sampled_data
