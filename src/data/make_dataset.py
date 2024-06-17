import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from src.utils.helpers import get_plot_dir, plot_data


def get_data(config):
    DATA = config.training_data

    def generate_gaussian_data(n_samples=1000, mean=[0, 0], cov=[[1, 0], [0, 1]]):
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return torch.tensor(data, dtype=torch.float32)

    def generate_anisotropic_single_gaussian(n_samples=1000):
        X = generate_gaussian_data(n_samples)
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

    def mnist_data():
        # create a transofrm to apply to each datapoint
        transform = transforms.Compose([transforms.ToTensor()])

        # download the MNIST datasets
        path = "~/datasets"
        train_dataset = MNIST(path, transform=transform, download=True)

        # create train dataloaders
        batch_size = 100
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        return train_loader

    if DATA == "normal":
        return generate_gaussian_data()
    elif DATA == "anisotropic":
        return generate_anisotropic_single_gaussian()
    elif DATA == "spiral":
        return generate_spiral_data()
    elif DATA == "mnist":
        return mnist_data()
    raise ValueError(f"Invalid data type {DATA}")


def plot_initial_data(data, config):
    plot_dir = get_plot_dir(config)
    plot_dir = plot_dir / "initial_data.png"
    plot_data(
        data,
        title="initial input data",
        x="Dimension 1",
        y="Dimension 2",
        path=plot_dir,
    )
