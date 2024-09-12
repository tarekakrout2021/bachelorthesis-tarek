import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.utils.helpers import load_model


def compute_log_likelihood(vae, x):
    total_log_likelihood = 0
    total_samples = 0
    for x_batch in x:
        x_batch = x_batch[0].reshape(-1, 784)
        batch_size = x_batch.size(0)

        mu, log_var = vae.encode(x_batch)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_sampled = vae.decode(z)

        px_given_z = MultivariateNormal(x_sampled, torch.eye(x_sampled.shape[-1]))

        log_likelihood = px_given_z.log_prob(x_batch).mean().item()

        total_log_likelihood += log_likelihood * batch_size
        total_samples += batch_size

    average_log_likelihood = total_log_likelihood / total_samples
    return average_log_likelihood


if __name__ == "__main__":
    models = ["baseline_mnist", "bitnet_mnist"]
    for model in models:
        vae_model = load_model(model)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        test_dataset = (
                MNIST(root="./data", train=False, transform=transform, download=True).data
                / 255.0
        )
        x = test_dataset.data[:1000]
        x = DataLoader(TensorDataset(x), batch_size=100, shuffle=False)

        log_likelihood = compute_log_likelihood(vae_model, x)
        print(f"{model}: log-likelihood: {log_likelihood}")
