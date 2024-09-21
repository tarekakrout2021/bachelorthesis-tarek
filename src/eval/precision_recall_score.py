import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from classifier import Net

from src.models.BitnetMnist import BitnetMnist
from src.models.VAE import VAE
from src.utils.helpers import load_model


def init_classifier() -> Net:
    classifier_path = "classifier.pth"
    classifier_state_dict = torch.load(
        classifier_path, map_location=torch.device("cpu")
    )
    classifier_model = Net()
    classifier_model.eval()
    classifier_model.load_state_dict(classifier_state_dict)
    return classifier_model


def calculate_conditional_precision(
    classifier_model: Net, vae_model: VAE, test_loader, plot_dir: Path
):
    """
    Compares the output of the classifier on the original data and the reconstructed data.
    Do the reconstructed samples resemble the original ones?
    """
    plot_data_recon, plot_data_init = [0] * 10, [0] * 10
    pred_original, pred_recon, true_labels = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            # prediction on the test_data
            output1 = classifier_model(data)  # output1 shape (btach_size, 10)
            pred1 = output1.data.max(1, keepdim=True)[1].reshape(
                1000
            )  # get index/label of the max element in the second dim
            pred_original.extend(np.round(pred1))

            # true labels
            true_labels.extend(target.data.view_as(pred1))

            # prediction on the reconstructed data
            x = data.view(1000, 784).to(
                vae_model.device
            )  # batch_size is 1000 and each image is 28x28
            recon_x, mu, logvar = vae_model(x)
            recon_x = recon_x.detach().cpu().reshape(1000, 1, 28, 28)
            output2 = classifier_model(recon_x)
            pred2 = output2.data.max(1, keepdim=True)[1].reshape(1000)
            pred_recon.extend(np.round(pred2))

    for label1, label2 in zip(true_labels, pred_original):
        plot_data_init[label1] += 1
        plot_data_recon[label2] += 1

    model_name = "bitnet" if isinstance(vae_model, BitnetMnist) else "baseline"
    plot_distribution(plot_data_init, f"Initial data distribution", plot_dir)
    plot_distribution(
        plot_data_recon, f"Reconstructed data distribution using {model_name}", plot_dir
    )

    pred_original = np.array(pred_original)
    pred_recon = np.array(pred_recon)
    print("precision of the classifier ", np.mean([pred_original == true_labels]))

    print("precision score:", np.mean([pred_original == pred_recon]))


def plot_distribution(freq: list, title: str, plot_dir: Path = None):
    """
    plots the frequency of each digit in the original and the reconstructed dataset
    """
    categories = [str(i) for i in range(10)]
    plt.bar(categories, freq)
    plt.xticks(np.arange(len(categories)), categories)
    plt.title(title, fontsize=18)
    plt.savefig(plot_dir / f"{title}.png")
    plt.close()


def plot_data_for_conditional_recall(
    classifier_model: Net, vae_model: VAE, test_loader, number: int, plot_dir: Path
):
    """
    plots images that got classified as <number> after reconstruction
    """
    recon_images = []
    original = []
    with torch.no_grad():
        for data, target in test_loader:
            # prediction on the reconstructed data
            x = data.view(1000, 784).to(
                vae_model.device
            )  # batch_size is 1000 and each image is 28x28
            recon_x, mu, logvar = vae_model(x)
            recon_x = recon_x.detach().cpu().reshape(1000, 1, 28, 28)
            output2 = classifier_model(recon_x)
            pred2 = output2.data.max(1, keepdim=True)[1].reshape(1000)
            indx = np.where([pred2 == number])[1]
            original.extend(data[indx])
            recon_images.extend(recon_x[indx])
    for i in range(10):
        plt.imshow(recon_images[i][0], cmap="gray")
        model_name = "bitnet" if isinstance(vae_model, BitnetMnist) else "baseline"
        number_dir = Path(plot_dir / f"recon_classified_as_{number}_{model_name}/")
        if not number_dir.exists():
            number_dir.mkdir()
        plt.savefig(number_dir / f"recon_{i}.png")
        plt.close()
        plt.imshow(original[i][0], cmap="gray")
        plt.savefig(number_dir / f"original_{i}.png")
        plt.close()


def unconditional_recall(classifier_model: Net, vae_model: VAE, plot_eval_dir: Path):
    freq = [0] * 10
    generated_samples = vae_model.sample(n_samples=1000)
    generated_samples = generated_samples.cpu().reshape(1000, 1, 28, 28)
    output = classifier_model(generated_samples)
    pred = output.data.max(1, keepdim=True)[1].reshape(1000)
    for label in pred:
        freq[label] += 1
    plot_distribution(freq, "Unconditional sampling distribution", plot_eval_dir)


def conditional_recall(
    classifier_model: Net, vae_model: VAE, number: int, plot_eval_dir: Path
):
    generated_samples = vae_model.sample(n_samples=1000)
    generated_samples = generated_samples.cpu().reshape(1000, 1, 28, 28)
    output = classifier_model(generated_samples)
    pred = output.data.max(1, keepdim=True)[1].reshape(1000)
    mask = pred == number
    number_images = generated_samples[mask]
    number_images = number_images[:10]
    for i in range(10):
        plt.imshow(number_images[i][0], cmap="gray")
        plt.savefig(plot_eval_dir / f"recon_classified_as_{number}_recon_{i}.png")
        plt.close()


def load_mnist_test_data() -> torch.utils.data.DataLoader:
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=1000,
        shuffle=True,
    )
    return test_loader


def main(plot_dir: Path, *args):
    # init classifier
    classifier_model = init_classifier()

    # load model
    model = load_model("bitnet_mnist")
    if args and args[0] == "baseline":
        model = load_model("baseline_mnist")

    plot_eval_dir = (
        Path("precision_recall_plots_bitnet")
        if "bitnet" in str(model)
        else Path("precision_recall_plots_baseline")
    )
    if not plot_eval_dir.exists():
        plot_eval_dir.mkdir()

    # load test data
    test_loader = load_mnist_test_data()
    unconditional_recall(classifier_model, model, plot_eval_dir)
    for i in range(10):
        conditional_recall(classifier_model, model, i, plot_eval_dir)

    calculate_conditional_precision(classifier_model, model, test_loader, plot_dir)
    test_loader.dataset.data = test_loader.dataset.data[:1000]
    for i in range(10):
        plot_data_for_conditional_recall(
            classifier_model, model, test_loader, i, plot_dir
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
