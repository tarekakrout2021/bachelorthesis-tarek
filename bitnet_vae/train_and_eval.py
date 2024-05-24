from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

from bitnet_vae.vae import VAE

DATA = "anisotropic"  # "normal" or "anisotropic"

PLOT_DIR = Path(f"../plots/{DATA}")
CHECKPOINT_DIR = Path("../checkpoints/")
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir(parents=True)

torch.manual_seed(0)


def get_data():
    def generate_gaussian_data(n_samples=1000, mean=[0, 0], cov=[[1, 0], [0, 1]]):
        data = np.random.multivariate_normal(mean, cov, n_samples)
        return torch.tensor(data, dtype=torch.float32)

    def generate_anisotropic_single_gaussian(n_samples=1000):
        X = generate_gaussian_data()
        transformation_matrix = np.array(
            [[5, 0], [0, 2]]
        )  # This creates an anisotropic effect
        rot_mat = np.array(
            [[np.sqrt(2) / 2, -np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2]]
        )
        transformation_matrix = transformation_matrix @ rot_mat
        data = np.dot(X, transformation_matrix)
        return torch.tensor(data, dtype=torch.float32)

    if DATA == "normal":
        return generate_gaussian_data()
    elif DATA == "anisotropic":
        return generate_anisotropic_single_gaussian()
    raise ValueError(f"Invalid data type {DATA}")


def plot_data(
    data,
    title="Input data",
    x="Dimension 1",
    y="Dimension 2",
    path=PLOT_DIR / "data.png",
):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.savefig(path)
    plt.close()


def train(model, optimizer, data_loader):
    mse_array = np.array([])
    kl_array = np.array([])
    training_array = np.array([])
    n_epochs = 150
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        mse_loss = 0
        kl_loss = 0
        for batch_idx, data in enumerate(data_loader):
            optimizer.zero_grad()
            data.to(model.device)
            recon_batch, mu, logvar = model(data)
            loss, mse, kl = model.loss_function(recon_batch, data, mu, logvar)
            train_loss += loss.item()
            mse_loss += mse.item()
            kl_loss += kl.item()
            loss.backward()
            optimizer.step()

        mse_array = np.append(mse_array, mse_loss)
        kl_array = np.append(kl_array, kl_loss)
        training_array = np.append(training_array, train_loss)
        print(f"Train Epoch: {epoch}  Loss: {train_loss :.6f}")

    # Plot loss
    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mse_array, label="Reconstruction Loss", color="blue")
    plt.plot(epochs, -kl_array, label="KL Divergence", color="red")
    plt.plot(epochs, training_array, label="Total Loss", color="black")
    plt.legend()
    plt.savefig(PLOT_DIR / "losses.png")
    plt.close()


def evaluate(model, data_loader):
    # Training mode: plot q(z|x) in training mode
    assert model.mode == "training"
    latent_variables = []
    for data in data_loader:
        mu, logvar = model.encode_latent(data)
        z = model.reparameterize(mu, logvar)
        latent_variables.append(z)
    latent_variables = torch.cat(latent_variables, 0)
    plot_dir = PLOT_DIR / "train_q(z|x).png"
    plot_data(
        latent_variables,
        title="Train: q(z|x)",
        x="Latent Dimension 1",
        y="Latent Dimension 2",
        path=plot_dir,
    )
    print(f"plotted at {plot_dir.absolute().resolve()}")

    model.change_to_inference()

    # Sanity checks
    print(model)  # Check whether all BitLinear layers are set to inference mode
    # Check whether weights are ternary
    # for name, param in list(model.named_parameters())[:2]:
    #     if param.requires_grad:
    #         print(name, param.data)

    # Inference mode: plot q(z|x) in inference mode
    assert model.mode == "inference"
    latent_variables = []
    for data in data_loader:
        mu, logvar = model.encode_latent(data)
        z = model.reparameterize(mu, logvar)
        latent_variables.append(z)
    latent_variables = torch.cat(latent_variables, 0)
    plot_data(
        latent_variables,
        title="Inference: q(z|x)",
        x="Latent Dimension 1",
        y="Latent Dimension 2",
        path=PLOT_DIR / "inference_q(z|x).png",
    )

    # Inference mode: sample from p(z) and decode
    assert model.mode == "inference"
    n_samples = 1000
    generated_data = model.sample(n_samples=n_samples, device="cpu")
    generated_data = generated_data.cpu().numpy()
    plot_data(
        generated_data,
        title="Inference: unconditional samples",
        x="Dimension 1",
        y="Dimension 2",
        path=PLOT_DIR / "inference_unconditional_samples.png",
    )

    # Inference mode: reconstruct data and plot
    assert model.mode == "inference"
    reconstructed_data = []
    for data in data_loader:
        mu, logvar = model.encode_latent(data)
        z = model.reparameterize(mu, logvar)
        reconstructions = model.decode(z)
        reconstructed_data.append(reconstructions.detach().cpu().numpy())
    reconstructed_data = np.concatenate(reconstructed_data, 0)
    plot_data(
        reconstructed_data,
        title="Inference: Reconstruction",
        x="Dimension 1",
        y="Dimension 2",
        path=PLOT_DIR / "inference_reconstruction.png",
    )


def main():
    data = get_data()
    plot_data(data)

    model = VAE()
    optimizer = Adam(model.parameters(), lr=3e-4)
    data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)

    train(model, optimizer, data_loader)
    evaluate(model, data_loader)


if __name__ == "__main__":
    main()
