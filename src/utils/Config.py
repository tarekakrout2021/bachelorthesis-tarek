from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data parameters
    train_data_path: str = field(default="data/train")
    training_data: str = field(default="anisotropic")
    seed: int = field(default=0)

    # Model parameters
    latent_dim: int = field(default=2)
    decoder_layers: List[int] = field(default_factory=lambda: [200, 200, 200])
    encoder_layers: List[int] = field(default_factory=lambda: [200, 200, 200])
    name: str = field(default="bitnet_synthetic")
    activation_layer: str = field(default="ReLU")
    norm: str = field(default="none")

    # Training parameters
    batch_size: int = field(default=64)  # maybe 10 is better ?
    epochs: int = field(default=10)
    learning_rate: float = field(default=0.001)
    run_id: str = field(default="0")
    device: str = field(default="cpu")
