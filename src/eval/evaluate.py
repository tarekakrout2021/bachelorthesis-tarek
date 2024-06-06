import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils import helpers


def evaluate(model, data_loader, config):
    model_name = config["model"]["name"]
    PLOT_DIR = helpers.get_plot_dir(config)

    if model_name == "bitnet_mnist":
        model.change_to_inference()

        assert model.mode == "inference"
        helpers.plot_latent_space(model)

        # Inference mode: reconstruct data
        assert model.mode == "inference"

        dataiter = iter(data_loader)
        image = next(dataiter)

        sample = 1
        x = image[0][sample, 0]

        plt.imshow(x, cmap="gray")
        plt.savefig(PLOT_DIR / "original_image.png")
        plt.close()

        x = x.view(1, 784).to(model.device)
        recon_x, mu, logvar = model(x)
        recon_x = recon_x.detach().cpu().reshape(28, 28)
        plt.imshow(recon_x, cmap="gray")

        plt.savefig(PLOT_DIR / "reconstructed_image.png")
        plt.close()

        # Inference mode: sample from prior
        assert model.mode == "inference"
        n_samples = 100
        generated_samples = model.sample(n_samples=n_samples, device=model.device)
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
        axes = axes.flatten()
        for ax, img in zip(axes, generated_samples):
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        plt.savefig(PLOT_DIR / f"sampled_from_the_prior.png")
        plt.close()

        # run weight stats
        assert model.mode == "inference"

        model.weight_stats()

        print(f"Stats: {model.n_0} zeros in ternary weights")
        print(f"Stats: {model.n_1} ones in ternary weights")
        print(f"Stats: {model.n_minus_1} minus ones in ternary weights")

        helpers.plot_bar(
            [model.n_minus_1 / 1e6, model.n_0 / 1e6, model.n_1 / 1e6],
            values=[-1, 0, 1],
            path=PLOT_DIR / "bar_chart_weights.png",
        )

    else:
        # Training mode: plot q(z|x) in training mode
        assert model.mode == "training"
        latent_variables = []
        for data in data_loader:
            mu, logvar = model.encode_latent(data)
            z = model.parameterize(mu, logvar)
            latent_variables.append(z)
        latent_variables = torch.cat(latent_variables, 0)
        plot_dir = PLOT_DIR / "train_q(z|x).png"
        helpers.plot_data(
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
            z = model.parameterize(mu, logvar)
            latent_variables.append(z)
        latent_variables = torch.cat(latent_variables, 0)
        helpers.plot_data(
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
        helpers.plot_data(
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
            z = model.parameterize(mu, logvar)
            reconstructions = model.decode(z)
            reconstructed_data.append(reconstructions.detach().cpu().numpy())
        reconstructed_data = np.concatenate(reconstructed_data, 0)
        helpers.plot_data(
            reconstructed_data,
            title="Inference: Reconstruction",
            x="Dimension 1",
            y="Dimension 2",
            path=PLOT_DIR / "inference_reconstruction.png",
        )

        # Inference mode: stat for ternary weights
        if model_name == "bitnet_vae":
            assert model.mode == "inference"
            model.weight_stats()

            # print(f"Stats: {model.n_0} zeros in ternary weights")
            # print(f"Stats: {model.n_1} ones in ternary weights")
            # print(f"Stats: {model.n_minus_1} minus ones in ternary weights")

            helpers.plot_bar(
                counts=[model.n_minus_1, model.n_0, model.n_1],
                values=[-1, 0, 1],
                path=PLOT_DIR / "ternary_weights.png",
            )
