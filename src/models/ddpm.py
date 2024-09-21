import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from src.models.Bitlinear158 import BitLinear158, BitLinear158Inference
from src.models.positional_embeddings import PositionalEmbedding


class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = BitLinear158(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()

        self.quantization_error: float = 0.0

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = (
            len(self.time_mlp.layer)
            + len(self.input_mlp1.layer)
            + len(self.input_mlp2.layer)
        )
        layers = [BitLinear158(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(BitLinear158(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

    def change_to_inference(self):
        """
        Replaces layers in network with inference layers and quantizes the weights.
        """

        def plot_heatmap(weights, quantized_weights, name, dir: str):
            """
            Plots the original weights and the quantized weights as heatmaps.
            """
            # Create directory for heatmaps
            plot_dir = Path(f"{dir}")
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
            plt.title(f"Non-Quantized Weights Layer \n {name}", fontsize=20)

            plt.subplot(1, 2, 2)
            plt.imshow(quantized_weights.to("cpu"), cmap="viridis")
            plt.colorbar()
            plt.title(f"Quantized Weights Layer \n {name}", fontsize=20)

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
                        "bitnet_heatmap_diffusion",
                    )
                else:
                    replace_bitlinear_layers(layer)

        replace_bitlinear_layers(self)
        self.mode = "inference"


class NoiseScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
