import matplotlib.pyplot as plt
import numpy as np
import torch

from src.models.BitnetSyntheticProbabilistic import BitnetSyntheticProbabilistic
from src.utils import helpers


def weight_stats(model, model_name, plot_dir, logger):
    if "bitnet" in model_name:
        # run weight stats
        model.weight_stats()

        logger.info(f"Stats: {model.n_0} zeros in ternary weights")
        logger.info(f"Stats: {model.n_1} ones in ternary weights")
        logger.info(f"Stats: {model.n_minus_1} minus ones in ternary weights")

        helpers.plot_bar(
            [model.n_minus_1, model.n_0, model.n_1],
            values=[-1, 0, 1],
            path=plot_dir / "bar_chart_weights.png",
        )
    else:
        helpers.plot_weight_distributions(model, plot_dir)


def plot_latent_space(data_loader, model, plot_dir, filename="train_q(z|x).png"):
    latent_variables = []
    for data in data_loader:
        mu, logvar = model.encode_latent(data)
        z = model.reparameterize(mu, logvar)
        latent_variables.append(z)
    latent_variables = torch.cat(latent_variables, 0)
    plot_dir = plot_dir / filename
    helpers.plot_data(
        latent_variables,
        title="Train: q(z|x)",
        x="Latent Dimension 1",
        y="Latent Dimension 2",
        path=plot_dir,
    )


def mnist_reconstruct_sample(data_loader, model, plot_dir, sample=1):
    dataiter = iter(data_loader)
    image = next(dataiter)

    x = image[0][sample, 0]

    plt.imshow(x, cmap="gray")
    plt.savefig(plot_dir / "original_image.png")
    plt.close()

    x = x.view(1, 784).to(model.device)
    recon_x, mu, logvar = model(x)
    recon_x = recon_x.detach().cpu().reshape(28, 28)
    plt.imshow(recon_x, cmap="gray")

    plt.savefig(plot_dir / "reconstructed_image.png")
    plt.close()


def mnist_sample_from_prior(model, plot_dir, n_samples=300):
    generated_samples = model.sample(n_samples=n_samples)
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    axes = axes.flatten()
    for ax, img in zip(axes, generated_samples):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.savefig(plot_dir / f"sampled_from_the_prior.png")
    plt.close()


def synthetic_sample_from_prior(model, plot_dir, n_samples=1000):
    generated_data = model.sample(n_samples=n_samples)
    generated_data = generated_data.cpu().numpy()
    helpers.plot_data(
        generated_data,
        title="Inference: unconditional samples",
        x="Dimension 1",
        y="Dimension 2",
        path=plot_dir / "inference_unconditional_samples.png",
    )


def synthetic_reconstruct_sample(data_loader, model, plot_dir):
    reconstructed_data = []
    if isinstance(model, BitnetSyntheticProbabilistic):
        for data in data_loader:
            mu, logvar = model.encode_latent(data)
            z = model.reparameterize(mu, logvar)
            recon_mu, recon_logvar = model.decode(z)
            recon_std = torch.sqrt(torch.exp(recon_logvar))
            eps = torch.randn_like(recon_std)
            reconstructions = recon_mu + eps * recon_std
            reconstructed_data.append(reconstructions.detach().cpu().numpy())
        reconstructed_data = np.concatenate(reconstructed_data, 0)
        helpers.plot_data(
            torch.tensor(reconstructed_data),
            title="Inference: Reconstruction",
            x="Dimension 1",
            y="Dimension 2",
            path=plot_dir / "inference_reconstruction.png",
        )
    else:
        for data in data_loader:
            mu, logvar = model.encode_latent(data)
            z = model.reparameterize(mu, logvar)
            reconstructions = model.decode(z)
            reconstructed_data.append(reconstructions.detach().cpu().numpy())
        reconstructed_data = np.concatenate(reconstructed_data, 0)
        helpers.plot_data(
            torch.tensor(reconstructed_data),
            title="Inference: Reconstruction",
            x="Dimension 1",
            y="Dimension 2",
            path=plot_dir / "inference_reconstruction.png",
        )


def evaluate_vae(model, data_loader, config, logger):
    model_name = config.model_name
    PLOT_DIR = helpers.get_plot_dir(config)

    if "mnist" in model_name:
        helpers.plot_weight_distributions(model, PLOT_DIR)

        # change to inference mode
        if "bitnet" in model_name:
            model.change_to_inference()
            assert model.mode == "inference"
        else:
            model.eval()

        # Sanity checks : Check whether all BitLinear layers are set to inference mode
        logger.debug(model)

        # Inference mode: reconstruct data
        mnist_reconstruct_sample(data_loader, model, PLOT_DIR)

        # Inference mode: sample from prior
        mnist_sample_from_prior(model, PLOT_DIR)

        # Inference mode
        weight_stats(model, model_name, PLOT_DIR, logger)

    else:
        # Training mode: plot q(z|x) in training mode
        plot_latent_space(data_loader, model, PLOT_DIR)
        helpers.plot_weight_distributions(model, PLOT_DIR)

        # Sanity checks : Check whether all BitLinear layers are set to inference mode
        if "bitnet" in model_name:
            model.change_to_inference()
            assert model.mode == "inference"
        else:
            model.eval()

        logger.debug(model)

        # Inference mode: plot q(z|x) in inference mode
        plot_latent_space(data_loader, model, PLOT_DIR, filename="inference_q(z|x).png")

        # Inference mode: sample from p(z) and decode
        synthetic_sample_from_prior(model, PLOT_DIR)

        # Inference mode: reconstruct data and plot
        synthetic_reconstruct_sample(data_loader, model, PLOT_DIR)

        # Inference mode
        weight_stats(model, model_name, PLOT_DIR, logger)


def evaluate_ddpm(model, data_loader, config, logger, run_dir):
    # does nothing for now ..
    pass


def evaluate(model, data_loader, config, logger, run_dir):
    if isinstance(config, helpers.VaeConfig):
        evaluate_vae(model, data_loader, config, logger)
    elif isinstance(config, helpers.DdpmConfig):
        evaluate_ddpm(model, data_loader, config, logger, run_dir)
