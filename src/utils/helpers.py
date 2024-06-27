import argparse
import logging
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.Baseline_synthetic import Baseline_synthetic
from src.models.Bitnet_mnist import Bitnet_mnist
from src.models.Bitnet_synthetic import Bitnet_synthetic
from src.utils.Config import Config


def plot_data(
        data,
        title="Input data",
        x="Dimension 1",
        y="Dimension 2",
        path="data.png",
):
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
    model_name = config.name

    if model_name == "baseline_synthetic":
        model = Baseline_synthetic(
            config.encoder_layers,
            config.decoder_layers,
            config.latent_dim,
        )
    elif model_name == "bitnet_synthetic":
        model = Bitnet_synthetic(
            config.encoder_layers,
            config.decoder_layers,
            config.latent_dim,
        )
    elif model_name == "bitnet_mnist":
        model = Bitnet_mnist(
            config.encoder_layers,
            config.decoder_layers,
            config.latent_dim,
        )
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return model


def plot_bar(counts, values=[-1, 0, 1], path="weights.png"):
    """
    Plot the distribution of weights.
    """
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
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(model.device)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
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


def init_logger(run_dir):
    logging.basicConfig(
        filename=run_dir / "test.log",
        filemode="a",
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG,
    )

    logger = logging.getLogger("logger")

    return logger


def log_model_info(logger, config):
    logging.info("Model configuration: ")
    logger.info(config)


def get_args():
    parser = argparse.ArgumentParser(description="Update YAML configuration.")
    parser.add_argument(
        "--model",
        help='Model name. Can be either "baseline_synthetic" or "bitnet_synthetic" or "bitnet_mnist".',
    )
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument("--latent_dim", type=int, help="Latent dimension.")
    parser.add_argument(
        "--training_data",
        help='can be either "normal" or "anisotropic" or "spiral" or "mnist".',
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

    parser.add_argument("--id", help="id of the run.")

    args = parser.parse_args()
    return args


def get_config(run_id):
    args = get_args()
    config = Config()
    config.run_id = run_id

    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.latent_dim:
        config.latent_dim = args.latent_dim
    if args.training_data:
        config.training_data = args.training_data
    if args.model:
        config.name = args.model
    if args.encoder_layers:
        config.encoder_layers = args.encoder_layers
    if args.decoder_layers:
        config.decoder_layers = args.decoder_layers
    # if there is a run_id in the arguments, use it.
    if args.id:
        config.run_id = args.id

    return config
