import argparse

import torch
from torch.optim import Adam
from train import train

from src.data.make_dataset import get_data, plot_initial_data
from src.eval.evaluate import evaluate
from src.utils.helpers import get_model, load_config, save_config, update_config


def main():
    parser = argparse.ArgumentParser(description="Update YAML configuration.")
    parser.add_argument(
        "--model",
        help='Model name. Can be either "baseline_synthetic" or "bitnet_synthetic" or "bitnet_mnist".',
    )
    parser.add_argument("--batch_size", type=int, help="Batch size.")
    parser.add_argument("--epochs", type=int, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, help="Learning rate.")
    parser.add_argument(
        "--training_data",
        help='can be either "normal" or "anisotropic" or "spiral" or "mnist".',
    )

    args = parser.parse_args()

    # TODO : maybe overcomplicated solution with config file ?
    config = load_config("./model_config.yaml")
    updated_config = update_config(config, args)
    save_config("./model_config.yaml", updated_config)

    config = updated_config

    data = get_data(config)
    if config["data"]["training_data"] != "mnist":
        plot_initial_data(data, config)

    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config["training"]["batch_size"], shuffle=True
    )
    # TODO  there is probably a better way to do this
    if args.training_data == "mnist":
        train(model, optimizer, data, config)
        evaluate(model, data, config)
    else:
        train(model, optimizer, data_loader, config)
        evaluate(model, data_loader, config)


if __name__ == "__main__":
    main()
