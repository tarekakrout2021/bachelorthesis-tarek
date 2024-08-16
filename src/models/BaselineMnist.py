import torch
import torch.nn as nn

from src.models.VAE import VAE
from src.utils.Config import VaeConfig


class BaselineMnist(VAE):
    def __init__(self, config: VaeConfig):
        super().__init__(
            config=config,
            layer=nn.Linear,
            input_dim=784,
        )
        self.decoder.append(nn.Sigmoid())

    def sample(self, n_samples=100):
        """
        Sample from p(z) and decode.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to("cpu")  # Sample from N(0, I)
            self.to("cpu")
            sampled_data = self.decode(z).cpu()

        sampled_data = sampled_data.view(n_samples, 28, 28)

        return sampled_data
