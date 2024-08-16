import textwrap
from pathlib import Path
from typing import List, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from src.models.Bitlinear158 import BitLinear158, BitLinear158Inference
from src.models.RMSNorm import RMSNorm
from src.utils.Config import VaeConfig


class VAE(nn.Module):
    def __init__(
        self,
        config: VaeConfig,
        layer: Type[nn.Linear] | Type[BitLinear158],
        input_dim: int = 2,
    ):
        super().__init__()
        self.decoder_layers: List[int] = [200, 200, 200]
        self.encoder_layers: List[int] = [200, 200, 200]
        if config.decoder_layers is not None:
            self.decoder_layers = config.decoder_layers

        if config.encoder_layers is not None:
            self.encoder_layers = config.encoder_layers

        self.input_dim: int = input_dim
        self.latent_dim: int = config.latent_dim

        # for stats
        self.n_0: int = 0
        self.n_1: int = 0
        self.n_minus_1: int = 0
        self.quantization_error: float = 0.0

        activation_layer: nn.Module = (
            nn.ReLU()
            if config.activation_layer == "ReLU"
            else nn.Sigmoid()
            if config.activation_layer == "Sigmoid"
            else nn.Tanh()
            if config.activation_layer == "tanh"
            else nn.ReLU()
        )

        # Encoder
        layers: List[nn.Module] = [
            layer(input_dim, self.encoder_layers[0]),
            activation_layer,
        ]
        if config.norm == "RMSNorm":
            layers.append(RMSNorm(self.encoder_layers[0]))
        for i in range(1, len(self.encoder_layers)):
            layers.append(layer(self.encoder_layers[i - 1], self.encoder_layers[i]))
            if config.norm == "RMSNorm":
                layers.append(RMSNorm(self.encoder_layers[i]))
            layers.append(activation_layer)
        self.encoder: nn.Sequential = nn.Sequential(*layers)

        self.mean_layer: nn.Module = layer(
            self.encoder_layers[-1], self.latent_dim
        )  # For mu
        self.log_var_layer: nn.Module = layer(
            self.encoder_layers[-1], self.latent_dim
        )  # For log variance

        # Decoder
        layers = [
            layer(self.latent_dim, self.decoder_layers[0]),
            activation_layer,
        ]
        if config.norm == "RMSNorm":
            layers.append(RMSNorm(self.decoder_layers[0]))
        for i in range(1, len(self.decoder_layers)):
            layers.append(layer(self.decoder_layers[i - 1], self.decoder_layers[i]))
            if config.norm == "RMSNorm":
                layers.append(RMSNorm(self.decoder_layers[i]))
            layers.append(activation_layer)
        layers.append(layer(self.decoder_layers[-1], input_dim))
        self.decoder: nn.Sequential = nn.Sequential(*layers)

        self.to(config.device)
        self.device: torch.device = torch.device(config.device)

        self.mode: str = "training"
        self.config: VaeConfig = config

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x.to(self.device)
        h: torch.Tensor = self.encoder(x)
        return self.mean_layer(h), self.log_var_layer(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std: torch.Tensor = torch.exp(0.5 * logvar).to(self.device)
        eps: torch.Tensor = torch.randn_like(std).to(self.device)
        return mu.to(self.device) + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z.to(self.device)
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu.to(self.device), logvar.to(self.device))
        return self.decode(z), mu, logvar

    def encode_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mu, logvar = self.encode(x)
            return mu, logvar

    @staticmethod
    def loss_function(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        config: VaeConfig,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        MSE: torch.Tensor = F.mse_loss(
            recon_x.to(config.device), x.to(config.device), reduction="sum"
        )
        KL: torch.Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KL, MSE, KL

    def change_to_inference(self):
        """
        Replaces layers in network with inference layers and quantizes the weights.
        """

        def plot_heatmap(weights, quantized_weights, name, config: VaeConfig):
            """
            Plots the original weights and the quantized weights as heatmaps.
            """
            # Create directory for heatmaps
            plot_dir = Path(f"runs/{config.run_id}/plots/heatmaps")
            if not plot_dir.exists():
                plot_dir.mkdir(parents=True)

            plt.subplot(1, 2, 2)
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(weights.to("cpu"), cmap="viridis")
            plt.colorbar()
            name = "\n".join(
                textwrap.wrap(name, width=40)
            )  # Adjust the width as needed
            plt.title(f"Non-Quantized Weights Layer \n {name}")

            plt.subplot(1, 2, 2)
            plt.imshow(quantized_weights.to("cpu"), cmap="viridis")
            plt.colorbar()
            plt.title(f"Quantized Weights Layer \n {name}")

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

                    # calculate the quantization error
                    self.quantization_error += (
                        torch.abs(old_layer_weight - new_layer.weight.data).sum().item()
                        / old_layer_weight.numel()
                    )

                    # Difference Plot
                    plot_heatmap(
                        old_layer_weight,
                        new_layer.weight.data,
                        f"{name}\n_{layer}",
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

    def sample(self, n_samples=100):
        """
        Sample from p(z) and decode.
        """
        self.eval()
        with torch.no_grad():
            # Sample from a standard normal distribution
            z = torch.randn(n_samples, self.latent_dim).to("cpu")
            # Decode the sample
            sampled_data = self.decode(z)
        return sampled_data
