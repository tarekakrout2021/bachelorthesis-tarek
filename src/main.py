import argparse
import shutil

import torch
from data.make_dataset import get_data, plot_initial_data
from eval.evaluate import evaluate
from torch.optim import Adam
from train import train
from utils.helpers import (
    generate_id,
    get_model,
    get_run_dir,
    load_config,
    save_config,
    update_config,
)


def main():
    torch.manual_seed(0)

    run_id = generate_id()
    print(f"Run ID: {run_id}")

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
    parser.add_argument(
        "--encoder_layers",
        type=int,
        nargs="+",
        help="array describing the encoder layer.",
    )
    parser.add_argument(
        "--decoder_layers",
        type=int,
        nargs="+",
        help="array describing the decoder layer.",
    )

    args = parser.parse_args()

    # TODO : maybe overcomplicated solution with config file ?
    config = load_config("./model_config.yaml")
    updated_config = update_config(config, args)
    updated_config["run_id"] = run_id
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

    run_dir = get_run_dir(config)
    torch.save(model.state_dict(), run_dir / "model.pth")
    shutil.copy("./model_config.yaml", run_dir / "config.yaml")


if __name__ == "__main__":
    main()
