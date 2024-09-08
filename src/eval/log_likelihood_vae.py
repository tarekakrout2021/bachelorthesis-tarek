import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

from src.utils.helpers import load_model


def compute_log_likelihood(vae, x, num_samples=5000):
    batch_size = x.size(0)

    # Flatten the input image x (because the decoder outputs a flat image)
    x_flat = x.view(batch_size, -1).float()

    mu, log_var = vae.encode(x)
    std = torch.exp(0.5 * log_var)

    # For Monte Carlo, draw `num_samples` samples from q(z|x)
    logs = []

    for _ in range(num_samples):
        # Sample z_i from q(z|x) using reparameterization
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Compute p(x|z_i) for the current z
        recon_x = vae.decode(z).float()  # Forward pass through the decoder

        # Reconstruction likelihood p(x|z)
        recon_x_flat = recon_x.view(batch_size, -1)

        # Compute the log likelihood log p(x|z). Using BCE because the data is Bernoulli distributed
        log_px_given_z = -F.binary_cross_entropy(
            recon_x_flat, x_flat, reduction="none"
        ).sum(dim=1)  # Sum over pixels

        # Compute log p(z_i), which is N(0, I)
        log_pz = -0.5 * torch.sum(
            z.pow(2) + torch.log(torch.tensor(2 * torch.pi)), dim=1
        )

        # Compute q(z_i|x) log probability
        log_qz_given_x = -0.5 * torch.sum(
            (z - mu).pow(2) / std.pow(2)
            + log_var
            + torch.log(torch.tensor([2 * torch.pi])),
            dim=1,
        )

        log = log_px_given_z + log_pz - log_qz_given_x
        logs.append(log)

    # Convert list to tensor: Shape: [batch_size, num_samples]
    logs_tensor = torch.stack(logs, dim=1)

    # Compute log p(x) using log-sum-exp trick for numerical stability
    log_likelihood = torch.logsumexp(logs_tensor, dim=1) - torch.log(
        torch.tensor(num_samples, dtype=torch.float)
    )

    return log_likelihood.mean()


if __name__ == "__main__":
    vae_model = load_model("bitnet_mnist")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    test_dataset = (
        MNIST(root="./data", train=False, transform=transform, download=True).data
        / 255.0
    )
    x = test_dataset.data[:1000].reshape(-1, 784)
    ret = compute_log_likelihood(vae_model, x, num_samples=100)
    print(ret)
