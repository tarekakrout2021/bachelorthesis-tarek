import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset


# TODO : some redundency in the code, need to refactor
def moons_dataset(n=8000):
    X, _ = make_moons(n_samples=n, random_state=42, noise=0.03)
    X[:, 0] = (X[:, 0] + 0.3) * 2 - 1
    X[:, 1] = (X[:, 1] + 0.3) * 3 - 1
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def line_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = rng.uniform(-0.5, 0.5, n)
    y = rng.uniform(-1, 1, n)
    X = np.stack((x, y), axis=1)
    X *= 4
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def circle_dataset(n=8000):
    rng = np.random.default_rng(42)
    x = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    y = np.round(rng.uniform(-0.5, 0.5, n) / 2, 1) * 2
    norm = np.sqrt(x**2 + y**2) + 1e-10
    x /= norm
    y /= norm
    theta = 2 * np.pi * rng.uniform(0, 1, n)
    r = rng.uniform(0, 0.03, n)
    x += r * np.cos(theta)
    y += r * np.sin(theta)
    X = np.stack((x, y), axis=1)
    X *= 3
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def dino_dataset(n=8000):
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
    return TensorDataset(torch.from_numpy(X.astype(np.float32)))


def generate_mixture_of_gaussians(n_samples: int = 9000):
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
    return TensorDataset(torch.from_numpy(data.astype(np.float32)))


def generate_circles(n_samples: int = 10_000, noise: float = 0.15):
    def circle(r, n):
        t = np.sqrt(np.random.rand(n)) * 2 * np.pi
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.vstack((x, y)).T + noise * np.random.randn(n, 2)

    c1 = circle(4, n_samples // 2)
    c2 = circle(10, n_samples // 2)
    res = np.vstack((c1, c2))
    return TensorDataset(torch.from_numpy(res.astype(np.float32)))


def get_dataset(name, n=8000):
    if name == "moons":
        return moons_dataset(n)
    elif name == "dino":
        return dino_dataset(n)
    elif name == "line":
        return line_dataset(n)
    elif name == "circle":
        return circle_dataset(n)
    elif name == "mixture":
        return generate_mixture_of_gaussians()
    elif name == "circles":
        return generate_circles()
    else:
        raise ValueError(f"Unknown dataset: {name}")
