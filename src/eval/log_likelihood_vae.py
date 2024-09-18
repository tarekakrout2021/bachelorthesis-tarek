import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST

from src.data.make_dataset import get_data_vae
from src.utils.helpers import load_model


def compute_log_likelihood(vae, data, num_samples: int = 100, data_dim: int = 784):
    total_log_likelihood = 0
    total_samples = 0

    for x_batch in data:
        logs = []
        x_batch = x_batch[0].reshape(-1, data_dim)
        batch_size = x_batch.size(0)

        mu, log_var = vae.encode(x_batch)
        std = torch.exp(0.5 * log_var)
        # Create a standard Normal prior p(z)
        prior = Normal(torch.zeros_like(mu), torch.ones_like(std))

        for _ in range(num_samples):
            eps = torch.randn_like(std)
            z = mu + eps * std
            x_sampled = vae.decode(z)

            # p(x|z) = N(mu(z), I)
            px_given_z = Normal(x_sampled, torch.ones_like(x_sampled))
            log_px_given_z = px_given_z.log_prob(x_batch).sum(dim=-1)

            # log p(z_i) is N(0, I)
            log_pz = prior.log_prob(z).sum(dim=-1)

            # q(z_i|x) log probability
            q_z_given_x = Normal(mu, std)
            log_qz_given_x = q_z_given_x.log_prob(z).sum(dim=-1)

            log = log_px_given_z + log_pz - log_qz_given_x
            logs.append(log)

        log_likelihoods = torch.stack(logs, dim=0)
        log_likelihood = torch.logsumexp(log_likelihoods, dim=0) - torch.log(
            torch.tensor(num_samples, dtype=torch.float)
        )

        total_log_likelihood += log_likelihood.sum().item()
        total_samples += batch_size

    average_log_likelihood = total_log_likelihood / total_samples
    return average_log_likelihood


if __name__ == "__main__":
    bitnet_models = ["baseline_mnist", "bitnet_mnist"]
    for model in bitnet_models:
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
        x = test_dataset.data[:10_000]
        x = DataLoader(TensorDataset(x), batch_size=100, shuffle=False)

        likelihood = compute_log_likelihood(vae_model, x, 100)
        print(f"{model} with MNIST: log-likelihood: {likelihood}")

    synthetic_data_models = ["baseline_synthetic", "bitnet_synthetic"]
    data = ["anisotropic", "spiral", "circles"]
    for model in synthetic_data_models:
        for data_type in data:
            vae_model, config = load_model(model, data_type)
            x: DataLoader = get_data_vae(config)
            likelihood = compute_log_likelihood(vae_model, x, 500, data_dim=2)
            print(f"{model} with {data_type}: log-likelihood: {likelihood}")
