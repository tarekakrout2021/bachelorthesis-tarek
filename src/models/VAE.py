from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.models.Bitlinear158 import BitLinear158, BitLinear158Inference
from src.utils.Config import Config


class VAE(nn.Module):
    def __init__(self, config: Config, layer, input_dim: int = 2):
        super().__init__()
        if config.decoder_layers is None:
            self.decoder_layers = [200, 200, 200]
        else:
            self.decoder_layers = config.decoder_layers

        if config.encoder_layers is None:
            self.encoder_layers = [200, 200, 200]
        else:
            self.encoder_layers = config.encoder_layers

        self.input_dim = input_dim
        self.latent_dim = config.latent_dim

        # for stats
        self.n_0 = 0
        self.n_1 = 0
        self.n_minus_1 = 0

        activation_layer = (
            nn.ReLU()
            if config.activation_layer == "ReLU"
            else nn.Sigmoid()
            if config.activation_layer == "Sigmoid"
            else nn.Tanh()
            if config.activation_layer == "tanh"
            else nn.ReLU()
        )

        # Encoder
        layers = [layer(input_dim, self.encoder_layers[0]), activation_layer]
        for i in range(1, len(self.encoder_layers)):
            layers.append(layer(self.encoder_layers[i - 1], self.encoder_layers[i]))
            layers.append(activation_layer)
        self.encoder = nn.Sequential(*layers)

        self.mean_layer = layer(self.encoder_layers[-1], self.latent_dim)  # For mu
        self.log_var_layer = layer(
            self.encoder_layers[-1], self.latent_dim
        )  # For log variance

        # Decoder
        # Decoder
        layers = [layer(self.latent_dim, self.decoder_layers[0]), activation_layer]
        for i in range(1, len(self.decoder_layers)):
            layers.append(layer(self.decoder_layers[i - 1], self.decoder_layers[i]))
            layers.append(activation_layer)
        layers.append(layer(self.decoder_layers[-1], input_dim))
        self.decoder = nn.Sequential(*layers)

        self.to(config.device)
        self.device = config.device

        self.mode = "training"
        self.config = config

    def encode(self, x):
        h = self.encoder(x)
        return self.mean_layer(h), self.log_var_layer(h)

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
        Replaces layers in network with inference layers and quantizes the weights.
        """

        def plot_heatmap(weights, quantized_weights, name, config: Config):
            plot_dir = Path(f"runs/{config.run_id}/plots/heatmaps")
            if not plot_dir.exists():
                plot_dir.mkdir(parents=True)

            plt.subplot(1, 2, 2)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(weights, cmap="viridis")
            plt.colorbar()
            plt.title(f"Non-Quantized Weights Layer {name}")

            plt.subplot(1, 2, 2)
            plt.imshow(quantized_weights, cmap="viridis")
            plt.colorbar()
            plt.title(f"Quantized Weights Layer {name}")

            plt.savefig(plot_dir / f"layer_{name}.png")
            plt.close()

        def replace_bitlinear_layers(module):
            for name, layer in module.named_children():
                if isinstance(layer, BitLinear158):
                    layer.beta = 1 / layer.weight.abs().mean().clamp(min=1e-5)

                    old_layer_weight = layer.weight.data.clone()

                    layer.weight = nn.Parameter(
                        (layer.weight * layer.beta).round().clamp(-1, 1)
                    )
                    layer.weight.detach()
                    layer.weight.requires_grad = False
                    new_layer = BitLinear158Inference(layer.input_dim, layer.output_dim)
                    new_layer.weight.data = layer.weight.data.clone()
                    new_layer.beta = layer.beta
                    setattr(module, name, new_layer)

                    # print("Layer: ", name)
                    # print(new_layer.weight.data - old_layer_weight)
                    # Difference Plot
                    plot_heatmap(
                        new_layer.weight.data,
                        old_layer_weight,
                        f"{name}_{layer}",
                        self.config,
                    )
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
