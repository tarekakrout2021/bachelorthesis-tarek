from dataclasses import dataclass, field
from typing import List


@dataclass
class VaeConfig:
    # Data parameters
    train_data_path: str = field(default="data/train")
    training_data: str = field(default="anisotropic")
    seed: int = field(default=0)

    # Model parameters
    latent_dim: int = field(default=2)
    decoder_layers: List[int] = field(default_factory=lambda: [200, 200, 200])
    encoder_layers: List[int] = field(default_factory=lambda: [200, 200, 200])
    model_name: str = field(default="bitnet_synthetic")
    activation_layer: str = field(default="ReLU")
    norm: str = field(default="none")

    # Training parameters
    batch_size: int = field(default=100)
    epochs: int = field(default=10)
    learning_rate: float = field(default=0.001)
    run_id: str = field(default="0")
    device: str = field(default="cpu")
    saving_interval: int = field(default=10)

    log_level: str = field(default="INFO")


@dataclass
class DdpmConfig:
    # Model parameters
    experiment_name: str = field(default="base")
    log_level: str = field(default="INFO")

    dataset: str = field(default="dino")

    # Training parameters
    epochs: int = field(default=10)
    train_batch_size: int = field(default=32)
    eval_batch_size: int = field(default=1000)
    num_epochs: int = field(default=200)
    learning_rate: float = field(default=1e-3)
    num_timesteps: int = field(default=50)
    beta_schedule: str = field(default="linear")
    embedding_size: int = field(default=128)
    hidden_size: int = field(default=128)
    hidden_layers: int = field(default=3)
    time_embedding: str = field(default="sinusoidal")
    input_embedding: str = field(default="sinusoidal")
    save_images_step: int = field(default=1)
