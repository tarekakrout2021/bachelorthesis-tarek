# import shutil
import torch
from data.make_dataset import get_data, plot_initial_data
from eval.evaluate import evaluate
from torch.optim import Adam
from train import train
from utils.helpers import *


def main():
    torch.manual_seed(0)

    run_id = generate_id()

    config = get_config(run_id)

    run_dir = get_run_dir(config)
    logger = init_logger(run_dir)
    logger.info(f"Run ID: {run_id}")
    print(run_id)

    data = get_data(config)
    if config.training_data != "mnist":
        plot_initial_data(data, config)

    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=config.batch_size, shuffle=True
    )
    # TODO  there is probably a better way to do this
    if config.training_data == "mnist":
        train(model, optimizer, data, config, logger)
        evaluate(model, data, config, logger)
    else:
        train(model, optimizer, data_loader, config, logger)
        evaluate(model, data_loader, config, logger)

    torch.save(model.state_dict(), run_dir / "model.pth")
    log_model_info(logger, config)


if __name__ == "__main__":
    main()
