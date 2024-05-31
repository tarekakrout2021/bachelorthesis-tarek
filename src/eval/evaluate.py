from pathlib import Path

import numpy as np
import torch

from src.utils.helpers import load_config, plot_data


def evaluate(model, data_loader):
    config = load_config("./model_config.yaml")
    model_name = config["model"]["name"]
    PLOT_DIR = Path(
        f"{config['output']['plot_dir']}/{model_name}/{config['data']['training_data']}"
    )

    if not PLOT_DIR.exists():
        PLOT_DIR.mkdir(parents=True)

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

    if model_name == "bitnet_vae":
        model.change_to_inference()

    # Sanity checks
    print(model)  # Check whether all BitLinear layers are set to inference mode
    # Check whether weights are ternary
    # for name, param in list(model.named_parameters())[:2]:
    #     if param.requires_grad:
    #         print(name, param.data)

    # Inference mode: plot q(z|x) in inference mode
    if model_name == "bitnet_vae":
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
    if model_name == "bitnet_vae":
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
    if model_name == "bitnet_vae":
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
