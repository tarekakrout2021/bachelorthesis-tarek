import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet_vae_mnist.bitlinear158 import BitLinear158, BitLinear158Inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=DEVICE):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            BitLinear158(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            BitLinear158(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            BitLinear158(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # latent mean and variance
        self.mean_layer = BitLinear158(latent_dim, 2)
        self.logvar_layer = BitLinear158(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            BitLinear158(2, latent_dim),
            nn.LeakyReLU(0.2),
            BitLinear158(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            BitLinear158(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            BitLinear158(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            BitLinear158(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        self.to(DEVICE)
        self.device = DEVICE

        self.mode = "training"

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def encode_latent(self, x):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            return mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # TODO: report MSE and KL seperately for debugging
        return MSE + KL, MSE, KL

    def change_to_inference(self):
        """
        Replaces layers in network with inference layers.
        """

        def replace_bitlinear_layers(module):
            for name, layer in module.named_children():
                if isinstance(layer, BitLinear158):
                    layer.beta = 1 / layer.weight.abs().mean().clamp(min=1e-5)
                    layer.weight = nn.Parameter(
                        (layer.weight * layer.beta).round().clamp(-1, 1)
                    )
                    layer.weight.detach()
                    layer.weight.requires_grad = False
                    new_layer = BitLinear158Inference(layer.input_dim, layer.output_dim)
                    new_layer.weight.data = layer.weight.data.clone()
                    new_layer.beta = layer.beta
                    setattr(module, name, new_layer)
                else:
                    replace_bitlinear_layers(layer)

        replace_bitlinear_layers(self)
        self.mode = "inference"

    def sample(self, n_samples=100, device="cpu"):
        """
        Sample from p(z) and decode.
        """
        with torch.no_grad():
            z = torch.randn(n_samples, 2).to(DEVICE)  # Sample from N(0, I)
            self.to(DEVICE)
            sampeled_data = self.decode(z).cpu()

        sampeled_data = sampeled_data.view(n_samples, 28, 28)

        return sampeled_data

    def get_number_of_parameters(self):
        return sum(p.numel() for p in self.parameters())
