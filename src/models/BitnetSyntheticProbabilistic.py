import torch
from torch import Tensor, nn

from src.models.VAE import VAE
from src.utils.Config import VaeConfig


class BitnetSyntheticProbabilistic(VAE):
    def __init__(self, config: VaeConfig):
        super().__init__(
            config=config,
            # layer=BitLinear158,
            layer=nn.Linear,
        )
        self.decoder.append(nn.ReLU())
        self.recon_mean_layer = nn.Linear(self.decoder_layers[-1], self.input_dim)
        self.recon_logvar_layer = nn.Linear(self.decoder_layers[-1], self.input_dim)

    def decode(self, z: torch.Tensor) -> tuple[Tensor, Tensor]:
        z.to(self.device)
        h = self.decoder(z)
        mu_dec, logvar_dec = self.recon_mean_layer(h), self.recon_logvar_layer(h)
        return mu_dec, logvar_dec

    def forward(self, x: torch.Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mu_enc, logvar_enc = self.encode(x)
        z = self.reparameterize(mu_enc.to(self.device), logvar_enc.to(self.device))
        mu_dec, logvar_dec = self.decode(z)
        return mu_dec, logvar_dec, mu_enc, logvar_enc

    def loss_function(self, recon_mu, recon_logvar, x, mu, logvar):
        # Reconstruction loss (Gaussian negative log-likelihood)
        recon_loss = 0.5 * torch.sum(
            recon_logvar + (x - recon_mu).pow(2) / torch.exp(recon_logvar)
        )

        # KL divergence between latent distribution and standard normal distribution
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_divergence, recon_loss, kl_divergence

    def sample(self, n_samples=100):
        """
        Sample from p(z) and decode.
        """
        self.eval()
        with torch.no_grad():
            # Sample from a standard normal distribution
            z = torch.randn(n_samples, self.latent_dim).to("cpu")
            # Decode the sample
            recon_mu, recon_logvar = self.decode(z)
            recon_std = torch.sqrt(torch.exp(recon_logvar))
            eps = torch.randn_like(recon_std)
            sampled_data = recon_mu + eps * recon_std

        return sampled_data
