import matplotlib.pyplot as plt
import yaml

from src.models.baseline_vae import VAE as baseline_vae_synthetic
from src.models.bitnet_vae import VAE as bitnet_vae_synthetic


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


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
    model_name = config["model"]["name"]

    if model_name == "baseline_vae":
        model = baseline_vae_synthetic()
    elif model_name == "bitnet_vae":
        model = bitnet_vae_synthetic()
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return model


def plot_bar(counts, values=[-1, 0, 1], path="weights.png"):
    plt.bar(values, counts, edgecolor="black")
    plt.title("Distribution of weights")
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.xticks(values)
    plt.savefig(path)
    plt.close()
