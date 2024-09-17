from typing import Callable, Dict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from src.data import datasets
from src.utils.Config import DdpmConfig, VaeConfig
from src.utils.helpers import get_plot_dir, plot_data


class SyntheticDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_data_loader(config: VaeConfig | DdpmConfig) -> DataLoader:
    if isinstance(config, VaeConfig):
        return get_data_vae(config)
    elif isinstance(config, DdpmConfig):
        dataset = datasets.get_dataset(config.dataset)
        data_loader = DataLoader(
            dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True
        )
        return data_loader


def get_data_vae(config: VaeConfig) -> DataLoader:
    def generate_gaussian_data(n_samples: int = 1000) -> DataLoader:
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return DataLoader(
            dataset=SyntheticDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
        )

    # 15_000
    def generate_mixture_of_gaussians(n_samples: int = 3000) -> DataLoader:
        mean1 = [0, 0]
        cov1 = [[1, 0], [0, 1]]
        data1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)
        mean2 = [5, 5]
        cov2 = [[1, 0], [0, 1]]
        data2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
        mean3 = [-10, 10]
        cov3 = [[0.5, 0], [0, 0.5]]
        data3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)
        data = np.vstack((data1, data2))
        data = np.vstack((data, data3))
        return DataLoader(
            SyntheticDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
        )

    def generate_moons(num_samples: int = 1_000, noise=0.1):
        x = make_moons(n_samples=num_samples, noise=noise)
        # make_moons returns a tuple with the first element being the data and the second being the labels
        # we don't need the labels, so we return the first element
        return DataLoader(
            SyntheticDataset(torch.tensor(x[0], dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
        )

    def generate_anisotropic_single_gaussian(n_samples: int = 1000) -> DataLoader:
        data_loader = generate_gaussian_data(n_samples)
        X: np.ndarray = data_loader.dataset.data.cpu().numpy()  # type: ignore

        transformation_matrix = np.array([[5, 0], [0, 2]])
        rot_mat = np.array(
            [[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]]
        )
        transformation_matrix = transformation_matrix @ rot_mat
        data = np.dot(X, transformation_matrix)
        return DataLoader(
            SyntheticDataset(torch.tensor(data, dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
        )

    def generate_spiral_data(n_samples: int = 10_000, noise: float = 0.5) -> DataLoader:
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
        r = 2 * theta + noise * np.random.randn(n_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        res = np.vstack((x, y)).T
        # return torch.tensor(res, dtype=torch.float32)
        return DataLoader(
            SyntheticDataset(torch.tensor(res, dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
        )

    def generate_circles(n_samples: int = 10_000, noise: float = 0.15) -> DataLoader:
        def circle(r, n):
            t = np.sqrt(np.random.rand(n)) * 2 * np.pi
            x = r * np.cos(t)
            y = r * np.sin(t)
            return np.vstack((x, y)).T + noise * np.random.randn(n, 2)

        c1 = circle(4, n_samples // 2)
        c2 = circle(10, n_samples // 2)
        res = np.vstack((c1, c2))
        return DataLoader(
            SyntheticDataset(torch.tensor(res, dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
        )

    def mnist_data() -> DataLoader:
        transform = transforms.Compose([transforms.ToTensor()])

        # download the MNIST datasets
        path = "~/datasets"
        train_dataset = MNIST(path, transform=transform, download=True)
        # train_dataset.data = train_dataset.data.float() / 255.0
        # create train dataloaders
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=config.batch_size, shuffle=True
        )

        return train_loader

    def dino_dataset(n=8000):
        # taken from this repo : https://github.com/tanelp/tiny-diffusion/tree/master
        df = pd.read_csv("data/static/dino.tsv", sep="\t")
        df = df[df["dataset"] == "dino"]

        rng = np.random.default_rng(42)
        ix = rng.integers(0, len(df), n)
        x = df["x"].iloc[ix].tolist()
        x = np.array(x) + rng.normal(size=len(x)) * 0.15
        y = df["y"].iloc[ix].tolist()
        y = np.array(y) + rng.normal(size=len(x)) * 0.15
        x = (x / 54 - 1) * 4
        y = (y / 48 - 1) * 4
        X = np.stack((x, y), axis=1)
        return DataLoader(
            SyntheticDataset(torch.tensor(X.astype(np.float32))),
            batch_size=config.batch_size,
            shuffle=True,
        )

    DataGenerator = Callable[[], DataLoader]

    data_generators: Dict[str, DataGenerator] = {
        "normal": generate_gaussian_data,
        "anisotropic": generate_anisotropic_single_gaussian,
        "spiral": generate_spiral_data,
        "mnist": mnist_data,
        "mixture": generate_mixture_of_gaussians,
        "moons": generate_moons,
        "circles": generate_circles,
        "dino": dino_dataset,
    }

    try:
        data_loader = data_generators[config.training_data]()
        return data_loader
    except KeyError:
        raise ValueError(f"Invalid data type {config.training_data}")


def plot_initial_data(data_loader: DataLoader, config: VaeConfig | DdpmConfig) -> None:
    plot_dir = get_plot_dir(config)
    plot_dir = plot_dir / "initial_data.png"
    plot_data(
        data_loader,
        title="initial input data",
        x="Dimension 1",
        y="Dimension 2",
        path=plot_dir,
    )
