from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.helpers import load_config


def train(model, optimizer, data_loader):
    config = load_config("./model_config.yaml")
    n_epochs = config["training"]["epochs"]
    model_name = config["model"]["name"]
    PLOT_DIR = Path(
        f"{config['output']['plot_dir']}/{model_name}/{config['data']['training_data']}"
    )

    if not PLOT_DIR.exists():
        PLOT_DIR.mkdir(parents=True)

    mse_array, kl_array, training_array = np.array([]), np.array([]), np.array([])
    for epoch in range(n_epochs):
        model.train()
        train_loss, mse_loss, kl_loss = 0, 0, 0
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
