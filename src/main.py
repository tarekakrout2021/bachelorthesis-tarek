# import shutil
from data.make_dataset import get_data_loader
from eval.evaluate import evaluate
from train.train import train
from utils.helpers import *


def main():
    torch.manual_seed(0)

    run_id = generate_id()

    config = get_config(run_id)

    run_dir = get_run_dir(config)
    logger = init_logger(run_dir, config.log_level)
    logger.info(f"Run ID: {run_id}")
    print(run_id)

    data_loader = get_data_loader(config)

    model = get_model(config)
    logger.debug(model)

    optimizer = get_optimizer(model, config)

    train(model, optimizer, data_loader, config, logger, run_dir)
    evaluate(model, data_loader, config, logger, run_dir)

    log_model_info(logger, config)


if __name__ == "__main__":
    main()
