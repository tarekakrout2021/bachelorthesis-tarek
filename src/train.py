import numpy as np

from src.utils.helpers import get_plot_dir, plot_loss


def train(model, optimizer, data_loader, config, logger):
    n_epochs = config.epochs
    model_name = config.name
    plot_dir = get_plot_dir(config)

    if model_name == "bitnet_mnist":
        n_data = data_loader.dataset.data.shape[0]

        mse_array, kl_array, training_array = np.array([]), np.array([]), np.array([])
        for epoch in range(n_epochs):
            model.train()
            train_loss = mse_loss = kl_loss = 0
            for batch_idx, (x, _) in enumerate(data_loader):
                # batch_size = 100 and x_dim = 784
                x = x.view(100, 784).to(model.device)
                optimizer.zero_grad()
                x.to(model.device)
                recon_batch, mu, logvar = model(x)
                loss, mse, kl = model.loss_function(recon_batch, x, mu, logvar, config)
                train_loss += loss.item()
                mse_loss += mse.item()
                kl_loss += kl.item()
                loss.backward()
                optimizer.step()

            mse_array = np.append(mse_array, mse_loss / n_data)
            kl_array = np.append(kl_array, kl_loss / n_data)
            training_array = np.append(training_array, train_loss / n_data)
            logger.info(
                f"Train Epoch: {epoch}  Average Loss: {train_loss / n_data:.6f}"
            )

        # Plot loss
        plot_loss(n_epochs, mse_array, kl_array, training_array, plot_dir)

    else:
        mse_array, kl_array, training_array = np.array([]), np.array([]), np.array([])
        for epoch in range(n_epochs):
            model.train()
            train_loss = mse_loss = kl_loss = 0
            for batch_idx, data in enumerate(data_loader):
                optimizer.zero_grad()
                data.to(model.device)
                recon_batch, mu, logvar = model(data)

                loss, mse, kl = model.loss_function(recon_batch, data, mu, logvar, config)

                train_loss += loss.item()
                mse_loss += mse.item()
                kl_loss += kl.item()

                loss.backward()
                optimizer.step()

            mse_array = np.append(mse_array, mse_loss)
            kl_array = np.append(kl_array, kl_loss)
            training_array = np.append(training_array, train_loss)
            logger.info(f"Train Epoch: {epoch}  Loss: {train_loss :.6f}")

        # Plot loss
        plot_loss(n_epochs, mse_array, kl_array, training_array, plot_dir)
