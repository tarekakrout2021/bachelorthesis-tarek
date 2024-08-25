import torch
import torch.nn as nn

from src.models.Bitlinear158 import BitLinear158
from src.models.VAE import VAE
from src.utils.Config import VaeConfig


class BitnetMnist(VAE):
    def __init__(self, config: VaeConfig):
        super().__init__(
            config=config,
            layer=BitLinear158,
            input_dim=784,
        )
        self.decoder.append(nn.Sigmoid())

    def sample(self, n_samples=100):
        """
        Sample from p(z) and decode.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(
                self.config.device
            )  # Sample from N(0, I)
            # self.to(self.config.device)
            sampled_data = self.decode(z).cpu()

        sampled_data = sampled_data.view(n_samples, 28, 28)

        return sampled_data
