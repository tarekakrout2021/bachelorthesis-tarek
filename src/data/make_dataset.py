import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from src.utils.Config import Config
from src.utils.helpers import get_plot_dir, plot_data


def get_data(config: Config):
    DATA = config.training_data

    def generate_gaussian_data(n_samples: int = 1000) -> torch.Tensor:
        mean = [0, 0]
        cov = [[1, 0], [0, 1]]
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return torch.tensor(data, dtype=torch.float32)

    def generate_mixture_of_gaussians(n_samples: int = 15_000) -> torch.Tensor:
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
        return torch.tensor(data, dtype=torch.float32)

    def generate_moons(num_samples: int = 1_000, noise=0.1):
        x = make_moons(n_samples=num_samples, noise=noise)
        # make_moons returns a tuple with the first element being the data and the second being the labels
        # we don't need the labels, so we return the first element
        return torch.tensor(x[0], dtype=torch.float32)

    def generate_anisotropic_single_gaussian(n_samples: int = 1000) -> torch.Tensor:
        X = generate_gaussian_data(n_samples)
        transformation_matrix = np.array([[5, 0], [0, 2]])
        rot_mat = np.array(
            [[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]]
        )
        transformation_matrix = transformation_matrix @ rot_mat
        data = np.dot(X, transformation_matrix)
        return torch.tensor(data, dtype=torch.float32)

    def generate_spiral_data(
        n_samples: int = 10_000, noise: float = 0.5
    ) -> torch.Tensor:
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi
        r = 2 * theta + noise * np.random.randn(n_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        res = np.vstack((x, y)).T
        return torch.tensor(res, dtype=torch.float32)

    def generate_circles(n_samples: int = 5_000, noise: float = 0.15) -> torch.Tensor:
        def circle(r, n):
            t = np.sqrt(np.random.rand(n)) * 2 * np.pi
            x = r * np.cos(t)
            y = r * np.sin(t)
            return np.vstack((x, y)).T + noise * np.random.randn(n, 2)

        c1 = circle(4, n_samples//2)
        c2 = circle(10, n_samples//2)
        res = np.vstack((c1, c2))
        return torch.tensor(res, dtype=torch.float32)

    def mnist_data() -> DataLoader:
        # create a transform to apply to each datapoint
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
    elif DATA == "mixture":
        return generate_mixture_of_gaussians()
    elif DATA == "moons":
        return generate_moons()
    elif DATA == "circles":
        return generate_circles()
    raise ValueError(f"Invalid data type {DATA}")


def plot_initial_data(data: torch.Tensor, config: Config) -> None:
    plot_dir = get_plot_dir(config)
    plot_dir = plot_dir / "initial_data.png"
    plot_data(
        data,
        title="initial input data",
        x="Dimension 1",
        y="Dimension 2",
        path=plot_dir,
    )
