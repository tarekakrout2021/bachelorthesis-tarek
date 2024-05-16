import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from bitlinear158 import BitLinear158, BitLinear158Inference


def generate_gaussian_data(n_samples=1000, mean=[0, 0], cov=[[1, 0], [0, 1]]):
    data = np.random.multivariate_normal(mean, cov, n_samples)
    return torch.tensor(data, dtype=torch.float32)


torch.manual_seed(0)
# data = torch.vstack((generate_gaussian_data(cov=[[1, 1],[1, 1]]), generate_gaussian_data(cov=[[1,-1],[-1,1]])))
# data = torch.vstack((generate_gaussian_data(mean=[1,1]), generate_gaussian_data(cov=[[1,-1],[-1,1]])))
# data = torch.vstack((generate_gaussian_data(mean=[10,10]) , generate_gaussian_data(mean=[-10,-10])))
data = generate_gaussian_data()
device = "cpu"


def plot_data(
    data,
    title="Scatter Plot of Gaussian Distributed Data",
    x="Dimension 1",
    y="Dimension 2",
    path="./plots/initial_data.png",
):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.savefig(path)
    plt.close()


plot_data(data)


class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=2, device=device):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = BitLinear158(input_dim, 200)
        self.fc2 = BitLinear158(200, 200)
        self.fc3 = BitLinear158(200, 200)
        self.fc31 = BitLinear158(200, latent_dim)  # For mu
        self.fc32 = BitLinear158(200, latent_dim)  # For log variance

        # Decoder
        self.decoder = nn.Sequential(
            BitLinear158(latent_dim, 100),
            nn.LeakyReLU(0.2),
            # RMSNorm(100),
            BitLinear158(100, 100),
            nn.LeakyReLU(0.2),
            # RMSNorm(100),
            # BitLinear158(100, 100),
            # nn.LeakyReLU(0.2),
            # RMSNorm(100),
            BitLinear158(100, input_dim),
        )

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc31(h3), self.fc32(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar).to(device)
        eps = torch.randn_like(std).to(device)
        mu.to(device)
        return mu + eps * std

    def decode(self, z):
        # TODO :
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode_latent(self, x):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            return mu, logvar


def loss_function(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # TODO: report MSE and KL seperately for debugging
    return MSE + KL


model = VAE().to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

# train the model
for epoch in range(50):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(data_loader):
        optimizer.zero_grad()
        data.to(device)
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Train Epoch: {epoch}  Loss: {loss.item() / len(data):.6f}")


latent_variables = []
for data in data_loader:
    mu, logvar = model.encode_latent(data)
    z = model.reparameterize(mu, logvar)
    latent_variables.append(z)
latent_variables = torch.cat(latent_variables, 0)


plot_data(
    latent_variables,
    title="Visualization of the Latent Space",
    x="Latent Dimension 1",
    y="Latent Dimension 2",
    path="./plots/latent_space.png",
)


def change_to_inference(model):
    bitlinear_layers = [
        (k, m) for k, m in model.named_modules() if type(m).__name__ == "BitLinear158"
    ]
    for name, layer in bitlinear_layers:
        layer.beta = 1 / layer.weight.abs().mean().clamp(min=1e-5)
        layer.weight = nn.Parameter((layer.weight * layer.beta).round().clamp(-1, 1))
        layer.weight.detach()
        layer.weight.requires_grad = False
        new_layer = BitLinear158Inference(layer.input_dim, layer.output_dim)
        new_layer.weight.data = layer.weight.data.clone()
        new_layer.beta = layer.beta
        setattr(model, name, new_layer)


change_to_inference(model)
change_to_inference(model.decoder)

print(model)


def sample_from_vae(model, n_samples=100, device="cpu"):
    model.eval()
    with torch.no_grad():
        # Sample from a standard normal distribution
        z = torch.randn(n_samples, model.latent_dim).to(device)
        # Decode the sample
        sampled_data = model.decode(z)
    return sampled_data


n_samples = 500
generated_data = sample_from_vae(model, n_samples=n_samples, device="cpu")

generated_data = generated_data.cpu().numpy()

plot_data(
    generated_data,
    title="Sampled Data Points from VAE",
    x="Dimension 1",
    y="Dimension 2",
    path="./plots/generated_data.png",
)
