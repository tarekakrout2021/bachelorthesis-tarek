import torch
from eval.evaluate import evaluate
from torch.optim import Adam
from train import train

from src.data.make_dataset import get_data, plot_initial_data
from src.utils.helpers import get_model, load_config


def main():
    data = get_data()
    plot_initial_data(data)

    config = load_config("./model_config.yaml")
    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config["training"]["learning_rate"])
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config["training"]["batch_size"], shuffle=True
    )

    train(model, optimizer, data_loader)
    evaluate(model, data_loader)


if __name__ == "__main__":
    main()
