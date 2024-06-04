import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bitlinear158 import BitLinear158, BitLinear158Inference

DEVICE = "cpu"


class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, num_layers=3, hidden_dim=200):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # for stats
        self.n_0 = 0
        self.n_1 = 0
        self.n_minus_1 = 0

        # Encoder
        encoder_layers = []
        encoder_layers.append(BitLinear158(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            encoder_layers.append(BitLinear158(hidden_dim, hidden_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc1 = BitLinear158(200, latent_dim)  # For mu
        self.fc2 = BitLinear158(200, latent_dim)  # For log variance

        # Decoder
        decoder_layers = []
        decoder_layers.append(BitLinear158(latent_dim, hidden_dim))
        decoder_layers.append(nn.LeakyReLU(0.2))
        for _ in range(num_layers - 1):
            decoder_layers.append(BitLinear158(hidden_dim, hidden_dim))
        decoder_layers.append(nn.LeakyReLU(0.2))
        decoder_layers.append(BitLinear158(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(DEVICE)
        self.device = DEVICE

        self.mode = "training"

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        mu.to(self.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode_latent(self, x):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            return mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="sum")
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
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

    def weight_stats(self):
        """
        counts the number of -1, 0, 1 weights in the network
        """

        def count(module):
            for name, layer in module.named_children():
                if isinstance(layer, BitLinear158Inference):
                    self.n_0 += (layer.weight == 0).sum().item()
                    self.n_1 += (layer.weight == 1).sum().item()
                    self.n_minus_1 += (layer.weight == -1).sum().item()
                else:
                    count(layer)

        count(self)

    def sample(self, n_samples=100, device="cpu"):
        """
        Sample from p(z) and decode.
        """
        self.eval()
        with torch.no_grad():
            # Sample from a standard normal distribution
            z = torch.randn(n_samples, self.latent_dim).to(device)
            # Decode the sample
            sampled_data = self.decode(z)
        return sampled_data
