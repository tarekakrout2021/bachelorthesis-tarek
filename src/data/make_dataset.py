from pathlib import Path

import numpy as np
import torch

from src.utils.helpers import load_config, plot_data

config = load_config("./model_config.yaml")
DATA = config["data"]["training_data"]
model_name = config["model"]["name"]
PLOT_DIR = Path(
    f"{config['output']['plot_dir']}/{model_name}/{config['data']['training_data']}"
)

if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)

torch.manual_seed(0)


def get_data():
    def generate_gaussian_data(n_samples=1000, mean=[0, 0], cov=[[1, 0], [0, 1]]):
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return torch.tensor(data, dtype=torch.float32)

    def generate_anisotropic_single_gaussian(n_samples=1000):
        X = generate_gaussian_data()
        transformation_matrix = np.array([[5, 0], [0, 2]])
        rot_mat = np.array(
            [[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]]
        )
        transformation_matrix = transformation_matrix @ rot_mat
        data = np.dot(X, transformation_matrix)
        return torch.tensor(data, dtype=torch.float32)

    def generate_spiral_data(n_samples=1000, noise=0.5):
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
        r = 2 * theta + noise * np.random.randn(n_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        res = np.vstack((x, y)).T
        return torch.tensor(res, dtype=torch.float32)

    if DATA == "normal":
        return generate_gaussian_data()
    elif DATA == "anisotropic":
        return generate_anisotropic_single_gaussian()
    elif DATA == "spiral":
        return generate_spiral_data()
    raise ValueError(f"Invalid data type {DATA}")


def plot_initial_data(data):
    plot_dir = PLOT_DIR / "initial_data.png"
    plot_data(
        data,
        title="initial input data",
        x="Dimension 1",
        y="Dimension 2",
        path=plot_dir,
    )
