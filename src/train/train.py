import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from tqdm.auto import tqdm

from src.data.make_dataset import plot_initial_data
from src.models.ddpm import NoiseScheduler
from src.utils.Config import DdpmConfig, VaeConfig
from src.utils.helpers import get_plot_dir, plot_loss, plot_quantization_error


def train(model, optimizer, data_loader, config, logger, run_dir):
    if isinstance(config, VaeConfig):
        if config.training_data != "mnist":
            plot_initial_data(data_loader, config)
        if config.model_name == "bitnet_synthetic_probabilistic":
            train_vae_probabilistic(
                model, optimizer, data_loader, config, logger, run_dir
            )
        else:
            train_vae(model, optimizer, data_loader, config, logger, run_dir)
    elif isinstance(config, DdpmConfig):
        train_ddpm(model, optimizer, data_loader, config, logger, run_dir)


def train_vae(
    model, optimizer, data_loader, config, logger, run_dir
):  # run_dir was used in the torch.save function
    n_epochs = config.epochs
    model_name = config.model_name
    plot_dir = get_plot_dir(config)

    n_data = data_loader.dataset.data.shape[0]
    logger.debug(f"Number of data points: {n_data}")

    mse_array, kl_array, training_array, quantization_array = (np.array([]),) * 4
    for epoch in range(n_epochs):
        model.train()
        train_loss = mse_loss = kl_loss = 0
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {epoch}")
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
            progress_bar.update(1)

        mse_array = np.append(mse_array, mse_loss / n_data)
        kl_array = np.append(kl_array, kl_loss / n_data)
        training_array = np.append(training_array, train_loss / n_data)
        logger.info(f"Train Epoch: {epoch}  Average Loss: {train_loss / n_data:.6f}")

        if epoch % config.saving_interval == 0 or epoch == n_epochs - 1:
            tmp_model = copy.deepcopy(model)
            tmp_model.change_to_inference()
            tmp_model.eval()
            quantization_array = np.append(
                quantization_array, tmp_model.quantization_error
            )
            logger.info(f"Quantization error: {tmp_model.quantization_error}")
            logger.info(f"Quantization array: {quantization_array}")
            # torch.save(tmp_model.state_dict(), run_dir / f"model_epoch_{epoch}.pth")

        progress_bar.close()

    # Plot loss
    plot_loss(n_epochs, mse_array, kl_array, training_array, plot_dir)
    plot_quantization_error(
        n_epochs, config.saving_interval, quantization_array, plot_dir
    )
    torch.save(model.state_dict(), run_dir / f"model.pth")


def train_vae_probabilistic(
    model, optimizer, data_loader, config, logger, run_dir
):  # run_dir was used in the torch.save function
    n_epochs = config.epochs
    model_name = config.model_name
    plot_dir = get_plot_dir(config)

    n_data = data_loader.dataset.data.shape[0]
    logger.debug(f"Number of data points: {n_data}")

    mse_array, kl_array, training_array, quantization_array = (np.array([]),) * 4
    for epoch in range(n_epochs):
        model.train()
        train_loss = mse_loss = kl_loss = 0
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {epoch}")
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
            recon_mu, recon_logvar, mu, logvar = model(data)

            # compute loss
            loss, mse, kl = model.loss_function(
                recon_mu, recon_logvar, data, mu, logvar
            )
            train_loss += loss.item()
            mse_loss += mse.item()
            kl_loss += kl.item()

            # backprop
            loss.backward()
            optimizer.step()
            progress_bar.update(1)

        mse_array = np.append(mse_array, mse_loss / n_data)
        kl_array = np.append(kl_array, kl_loss / n_data)
        training_array = np.append(training_array, train_loss / n_data)
        logger.info(f"Train Epoch: {epoch}  Average Loss: {train_loss / n_data:.6f}")

        if epoch % config.saving_interval == 0 or epoch == n_epochs - 1:
            tmp_model = copy.deepcopy(model)
            tmp_model.change_to_inference()
            tmp_model.eval()
            quantization_array = np.append(
                quantization_array, tmp_model.quantization_error
            )
            logger.info(f"Quantization error: {tmp_model.quantization_error}")
            logger.info(f"Quantization array: {quantization_array}")
            # torch.save(tmp_model.state_dict(), run_dir / f"model_epoch_{epoch}.pth")

        progress_bar.close()

    # Plot loss
    plot_loss(n_epochs, mse_array, kl_array, training_array, plot_dir)
    plot_quantization_error(
        n_epochs, config.saving_interval, quantization_array, plot_dir
    )
    torch.save(model.state_dict(), run_dir / f"model.pth")


def train_ddpm(model, optimizer, data_loader, config, logger, run_dir):
    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps, beta_schedule=config.beta_schedule
    )

    global_step = 0
    frames = []
    losses = []
    quantization_array = np.array([])
    logger.info("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(data_loader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(data_loader):
            batch = batch[0]
            noise = torch.randn(batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if epoch % config.save_images_step == 0 or epoch == config.num_epochs - 1:
            # generate data with the model to later visualize the learning process
            tmp_model = copy.deepcopy(model)
            tmp_model.eval()
            tmp_model.change_to_inference()

            sample = torch.randn(config.eval_batch_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
                with torch.no_grad():
                    residual = tmp_model(sample, t)
                sample = noise_scheduler.step(residual, t[0], sample)
            frames.append(sample.numpy())

            quantization_array = np.append(
                quantization_array, tmp_model.quantization_error
            )

    logger.debug(model)
    model.eval()
    model.change_to_inference()
    logger.debug(model)

    logger.info("Saving model...")
    outdir = f"{run_dir}/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model.pth")

    logger.info("Saving images...")
    imgdir = f"{outdir}/images"
    os.makedirs(imgdir, exist_ok=True)
    frames = np.stack(frames)
    for i, frame in enumerate(frames):
        plt.figure(figsize=(10, 10))
        plt.scatter(frame[:, 0], frame[:, 1])
        plt.xlim(-6, 6)
        plt.ylim(-6, 6)
        plt.savefig(f"{imgdir}/{i:04}.png")
        plt.close()

    plot_quantization_error(
        config.num_epochs, config.save_images_step, quantization_array, Path(outdir)
    )

    logger.info("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    logger.info("Saving frames...")
    np.save(f"{outdir}/frames.npy", frames)
