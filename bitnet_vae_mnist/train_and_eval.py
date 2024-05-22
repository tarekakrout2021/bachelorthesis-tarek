from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from bitnet_vae_mnist.vae import VAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)
DATA = "mnist"

PLOT_DIR = Path(f"../plots/{DATA}")
CHECKPOINT_DIR = Path("../checkpoints/")
if not PLOT_DIR.exists():
    PLOT_DIR.mkdir(parents=True)
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir(parents=True)

torch.manual_seed(0)


def get_data():
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

    return train_loader, train_dataset


def visualize_data(train_loader):
    dataiter = iter(train_loader)
    image = next(dataiter)

    num_samples = 25
    sample_images = [image[0][i, 0] for i in range(num_samples)]

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

    for ax, im in zip(grid, sample_images):
        ax.imshow(im, cmap="gray")
        ax.axis("off")
    plt.savefig(PLOT_DIR / "initial_data.png")
    plt.close()


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
    plt.savefig(path)
    plt.close()


def train(model, optimizer, data_loader):
    model.to(DEVICE)

    n_data = data_loader.dataset.data.shape[0]

    mse_array = np.array([])
    kl_array = np.array([])
    training_array = np.array([])
    n_epochs = 50
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        mse_loss = 0
        kl_loss = 0
        for batch_idx, (x, _) in enumerate(data_loader):
            # batch_size = 100 and x_dim = 784
            x = x.view(100, 784).to(DEVICE)
            optimizer.zero_grad()
            x.to(model.device)
            recon_batch, mu, logvar = model(x)
            loss, mse, kl = model.loss_function(recon_batch, x, mu, logvar)
            train_loss += loss.item()
            mse_loss += mse.item()
            kl_loss += kl.item()
            loss.backward()
            optimizer.step()

        mse_array = np.append(mse_array, mse_loss / n_data)
        kl_array = np.append(kl_array, kl_loss / n_data)
        training_array = np.append(training_array, train_loss / n_data)
        print(f"Train Epoch: {epoch}  Average Loss: {train_loss / n_data:.6f}")

    # Plot loss
    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mse_array, label="Reconstruction Loss", color="blue")
    plt.plot(epochs, -kl_array, label="KL Divergence", color="red")
    plt.plot(epochs, training_array, label="Total Loss", color="black")
    plt.legend()
    plt.savefig(PLOT_DIR / "losses.png")
    plt.close()

    # Save the model's state_dict
    torch.save(
        model.state_dict(), CHECKPOINT_DIR / "bitnet_vae_mnist_more_layers_4_hidden.pth"
    )
    print("Model saved.")


# for debugging
def generate_digit(model, mean, var):
    z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(DEVICE)
    x_decoded = model.decode(z_sample)
    digit = x_decoded.detach().cpu().reshape(28, 28)  # reshape vector to 2d array
    plt.imshow(digit, cmap="gray")
    plt.axis("off")
    plt.show()


def plot_latent_space(model, scale=1.0, n=25, digit_size=28, figsize=15):
    # display a n*n 2D manifold of digits
    figure = np.zeros((digit_size * n, digit_size * n))

    # construct a grid
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    model.to(DEVICE)

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(DEVICE)
            x_decoded = model.decode(z_sample)
            digit = x_decoded[0].detach().cpu().reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.title("VAE Latent Space Visualization")
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("mean, z [0]")
    plt.ylabel("var, z [1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(PLOT_DIR / f"mnist_reconstructed_{scale}.png")
    plt.close()


def evaluate(model, data_loader):
    model.change_to_inference()

    assert model.mode == "inference"
    plot_latent_space(model)

    # Inference mode: reconstruct data
    assert model.mode == "inference"

    dataiter = iter(data_loader)
    image = next(dataiter)

    sample = 1
    x = image[0][sample, 0]

    plt.imshow(x, cmap="gray")
    plt.savefig(PLOT_DIR / "original_image.png")

    x = x.view(1, 784).to(DEVICE)
    recon_x, mu, logvar = model(x)
    recon_x = recon_x.detach().cpu().reshape(28, 28)
    plt.imshow(recon_x, cmap="gray")

    plt.savefig(PLOT_DIR / "reconstructed_image.png")

    # Inference mode: sample from prior
    assert model.mode == "inference"
    n_samples = 100
    generated_samples = model.sample(n_samples=n_samples, device=DEVICE)
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    axes = axes.flatten()
    for ax, img in zip(axes, generated_samples):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.savefig(PLOT_DIR / f"sampled_from_the_prior.png")
    plt.close()


def main():
    data_loader, data_set = get_data()
    visualize_data(data_loader)

    model = VAE()
    optimizer = Adam(model.parameters(), lr=1e-3)

    train(model, optimizer, data_loader)
    evaluate(model, data_loader)

    # print(model.get_number_of_parameters(), 'parameters')


if __name__ == "__main__":
    main()
