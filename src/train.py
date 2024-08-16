import copy
import logging

import numpy as np
import torch

from src.utils.helpers import get_plot_dir, plot_loss, plot_data, plot_quantization_error


def train(model, optimizer, data_loader, config, logger, run_dir):  # run_dir was used in the torch.save function
    n_epochs = config.epochs
    model_name = config.model_name
    plot_dir = get_plot_dir(config)

    n_data = data_loader.dataset.data.shape[0]
    logger.debug(f"Number of data points: {n_data}")

    mse_array, kl_array, training_array, quantization_array = (np.array([]),) * 4
    for epoch in range(n_epochs):
        model.train()
        train_loss = mse_loss = kl_loss = 0
        for batch_idx, data in enumerate(data_loader):
            if "mnist" in model_name:
                x = data[0]
                x = x.view(config.batch_size, 784).to(
                    model.device
                )  # x_dim = 784 = 28 * 28
                data = x
            optimizer.zero_grad()
            data.to(model.device)

            # forward pass
            recon_batch, mu, logvar = model(data)

            # compute loss
            loss, mse, kl = model.loss_function(recon_batch, data, mu, logvar, config)
            train_loss += loss.item()
            mse_loss += mse.item()
            kl_loss += kl.item()

            # backprop
            loss.backward()
            optimizer.step()

        mse_array = np.append(mse_array, mse_loss / n_data)
        kl_array = np.append(kl_array, kl_loss / n_data)
        training_array = np.append(training_array, train_loss / n_data)
        logger.info(f"Train Epoch: {epoch}  Average Loss: {train_loss / n_data:.6f}")

        if epoch % config.saving_interval == 0 or epoch == n_epochs - 1:
            tmp_model = copy.deepcopy(model)
            tmp_model.change_to_inference()
            tmp_model.eval()
            quantization_array = np.append(quantization_array, tmp_model.quantization_error)
            logger.info(f"Quantization error: {tmp_model.quantization_error}")
            logger.info(f"Quantization array: {quantization_array}")
            # torch.save(tmp_model.state_dict(), run_dir / f"model_epoch_{epoch}.pth")

    # Plot loss
    plot_loss(n_epochs, mse_array, kl_array, training_array, plot_dir)
    plot_quantization_error(n_epochs, config.saving_interval, quantization_array, plot_dir)
