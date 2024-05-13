import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Bitlinear158 import BitLinear158, BitLinear158Inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VAE(nn.Module):
    def __init__(
        self,
        layer,
        activation_layer,
        encoder_layers=None,
        decoder_layers=None,
        input_dim=2,
        latent_dim=2,
    ):
        super().__init__()
        if decoder_layers is None:
            decoder_layers = [200, 200, 200]
        if encoder_layers is None:
            encoder_layers = [200, 200, 200]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # for stats
        self.n_0 = 0
        self.n_1 = 0
        self.n_minus_1 = 0

        # Encoder
        layers = [layer(input_dim, encoder_layers[0]), activation_layer]
        for i in range(1, len(encoder_layers)):
            layers.append(layer(encoder_layers[i - 1], encoder_layers[i]))
            layers.append(activation_layer)
        self.encoder = nn.Sequential(*layers)

        self.mean_layer = layer(encoder_layers[-1], latent_dim)  # For mu
        self.log_var_layer = layer(encoder_layers[-1], latent_dim)  # For log variance

        # Decoder
        layers = [layer(latent_dim, decoder_layers[0]), activation_layer]
        for i in range(1, len(decoder_layers)):
            layers.append(layer(decoder_layers[i - 1], decoder_layers[i]))
            layers.append(activation_layer)
        layers.append(layer(decoder_layers[-1], input_dim))
        self.decoder = nn.Sequential(*layers)

        self.to(DEVICE)
        self.device = DEVICE

        self.mode = "training"

    def encode(self, x):
        h = self.encoder(x)
        return self.mean_layer(h), self.log_var_layer(h)

    def parameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(self.device)
        eps = torch.randn_like(std).to(self.device)
        mu.to(self.device)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.parameterize(mu, logvar)
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
        Replaces layers in network with inference layers and quantizes the weights.
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
        counts the number of -1, 0, 1 weights in the network.
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
