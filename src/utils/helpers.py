import argparse
import logging
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.models.BaselineMnist import BaselineMnist
from src.models.BaselineSynthetic import BaselineSynthetic
from src.models.BitnetMnist import BitnetMnist
from src.models.BitnetSynthetic import BitnetSynthetic
from src.utils.Config import Config


def plot_data(
    data: DataLoader | torch.Tensor,
    title="Input data",
    x="Dimension 1",
    y="Dimension 2",
    path="data.png",
):
    if isinstance(data, DataLoader):
        data = data.dataset.data.cpu().numpy()
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.axis("equal")
    plt.savefig(path)
    plt.close()


def get_model(config):
    model_name = config.model_name

    if model_name == "baseline_synthetic":
        model = BaselineSynthetic(config)
    elif model_name == "bitnet_synthetic":
        model = BitnetSynthetic(config)
    elif model_name == "bitnet_mnist":
        model = BitnetMnist(config)
    elif model_name == "baseline_mnist":
        model = BaselineMnist(config)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return model


def plot_bar(counts, values=None, path="weights.png"):
    """
    Plot the distribution of weights.
    """
    if values is None:
        values = [-1, 0, 1]
    plt.figure(figsize=(8, 6))
    plt.bar(values, counts, edgecolor="black")
    plt.title("Distribution of weights")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.xticks(values)
    plt.savefig(path)
    plt.close()


def plot_latent_space(model, config, scale=1.0, n=25, digit_size=28, figsize=15):
    """
    Plot the latent space of the VAE model. Only for mnist data.
    """
    plot_dir = get_plot_dir(config)

    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float)
            if config.device == "cuda":
                z_sample = z_sample.cuda()

            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title("VAE Latent Space Visualization")
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(plot_dir / f"mnist_reconstructed_{scale}.png")
    plt.close()


def get_plot_dir(config):
    """
    returns the plot directory and creates it if it does not exist.
    """
    plot_dir = Path(f"runs/{config.run_id}/plots")
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    return plot_dir


def get_run_dir(config):
    """
    returns the run directory and creates it if it does not exist.
    """
    run_dir = Path(f"runs/{config.run_id}")
    if not run_dir.exists():
        run_dir.mkdir(parents=True)

    return run_dir


def plot_loss(n_epochs, mse_array, kl_array, training_array, plot_dir):
    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mse_array, label="Reconstruction Loss", color="blue")
    plt.plot(epochs, -kl_array, label="KL Divergence", color="red")
    plt.plot(epochs, training_array, label="Total Loss", color="black")
    plt.legend()
    plt.savefig(plot_dir / "losses.png")
    plt.close()


def generate_id():
    return str(uuid.uuid4())


def init_logger(run_dir, log_level):
    logging.basicConfig(
        filename=run_dir / "test.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("logger")
    logger.setLevel(getattr(logging, log_level.upper(), None))
    return logger


def log_model_info(logger, config):
    logging.debug("Model configuration: ")
    logger.debug(config)


def plot_weight_distributions(model, plot_dir):
    for name, param in model.named_parameters():
        if "weight" in name:  # filter out biases or other parameters
            plt.figure(figsize=(8, 6))
            sns.histplot(param.data.cpu().numpy().flatten(), bins=50, kde=True)
            plt.title(f"Weight Distribution of {name}")
            plt.xlabel("Weight values")
            plt.ylabel("Frequency")
            plt.savefig(plot_dir / "weight_distribution.png")
            plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Update YAML configuration.")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=[
            "baseline_synthetic",
            "bitnet_synthetic",
            "bitnet_mnist",
            "baseline_mnist",
        ],
        default="bitnet_synthetic",
    )
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--latent_dim", type=int, help="Latent dimension.")
    parser.add_argument(
        "--activation_layer",
        type=str,
        choices=["ReLU", "Sigmoid", "tanh"],
        default="ReLU",
    )
    parser.add_argument(
        "--training_data",
        type=str,
        choices=[
            "normal",
            "anisotropic",
            "spiral",
            "mnist",
            "dino",
            "moons",
            "circles",
            "mixture",
        ],
        default="spiral",
    )
    parser.add_argument(
        "--encoder_layers",
        type=int,
        nargs="+",
        help="array describing the encoder layer.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        nargs="+",
        help="array describing the decoder layer.",
    )

    parser.add_argument("--run_id", help="id of the run.")

    parser.add_argument("--norm", help="if it uses RMSNorm or not.")

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="device to run the model.",
    )

    parser.add_argument(
        "--saving_interval", type=int, help="Interval to save the model."
    )

    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Log level for the logger.",
    )

    args = parser.parse_args()
    return args


def get_config(run_id):
    args = get_args()
    config = Config()
    config.run_id = run_id

    for arg in vars(args):
        if getattr(args, arg) is not None:
            setattr(config, arg, getattr(args, arg))

    return config
