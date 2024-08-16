# BitNet VAE

## Setup
### Commit Hook
On your local machine:
```bash
pip install pre-commit
pre-commit install
pip install ruff
pre-commit run --all-files  # to run manually before a commit
```

### Container
From the repo root:
```bash
apptainer build python_container.sif python_container.def
```

### Run the code
From the root folder:
```bash
export PYTHONPATH=$PYTHONPATH:`pwd`
```

### Training a VAE Model

Use the `main.py` script to train a VAE model with various configurable options.

#### Basic Command

```bash
python main.py train_vae [OPTIONS]
```

Available Options
Model Configuration:

    --model_name {baseline_synthetic, bitnet_synthetic, bitnet_mnist, baseline_mnist}
        Specifies the model architecture to be used.

Training Parameters:

    --batch_size BATCH_SIZE
        Batch size for training (default: 64).
    --epochs EPOCHS
        Number of epochs to train (default: 100).
    --learning_rate LEARNING_RATE
        Learning rate for the optimizer (default: 0.001).

Model Architecture:

    --latent_dim LATENT_DIM
        Dimensionality of the latent space (default: 128).
    --activation_layer {ReLU, Sigmoid, tanh}
        Activation function used in the network (default: ReLU).
    --encoder_layers ENCODER_LAYERS [ENCODER_LAYERS ...]
        Sizes of the layers in the encoder (e.g., 256 128).
    --decoder_layers DECODER_LAYERS [DECODER_LAYERS ...]
        Sizes of the layers in the decoder (e.g., 128 256).

Data:

    --training_data {normal, anisotropic, spiral, mnist, dino, moons, circles, mixture}
        Specifies the dataset to be used for training.

Experiment Tracking:

    --run_id RUN_ID
        Identifier for the current experiment run.

Normalization:

    --norm NORM
        Type of normalization to use (e.g., batch_norm, layer_norm).

Device:

    --device {cpu, cuda}
        Specifies whether to run the training on CPU or CUDA-enabled GPU.

Logging & Checkpoints:

    --saving_interval SAVING_INTERVAL
        Interval (in epochs) to save model checkpoints (default: 10).
    --log_level {DEBUG, INFO, WARNING, ERROR, CRITICAL}
        Logging level for the experiment (default: INFO).

#### Example Command
```bash
python main.py train_vae --model_name baseline_mnist --batch_size 128 --epochs 50 --learning_rate 0.0005 \
                         --latent_dim 64 --activation_layer ReLU --training_data mnist \
                         --encoder_layers 512 256 --decoder_layers 256 512 \
                         --device cuda --saving_interval 5 --log_level INFO
```

Models are named as `{model}_{dataset}` where dataset is either `synthetic` or `mnist` and  model is either `baseline` or `bitnet`.
Training data can either be `anisotropic`, `normal`, `spiral` or `mnist`.

### Training the diffusion model
#### Basic Command

```bash
python main.py train_ddpm [OPTIONS]
```
Available Options
Training Parameters:

    --epochs EPOCHS
        Number of epochs to train (default: 100).
    --train_batch_size TRAIN_BATCH_SIZE
        Batch size for training (default: 64).
    --eval_batch_size EVAL_BATCH_SIZE
        Batch size for evaluation (default: 64).
    --learning_rate LEARNING_RATE
        Learning rate for the optimizer (default: 0.001).
    --num_epochs NUM_EPOCHS
        Total number of epochs for training.

Experiment Configuration:

    --experiment_name EXPERIMENT_NAME
        Name for the current experiment run.
    --dataset {circle, dino, line, moons}
        Specifies the dataset to be used for training.

DDPM-Specific Parameters:

    --num_timesteps NUM_TIMESTEPS
        Number of timesteps for the diffusion process (default: 1000).
    --beta_schedule {linear, quadratic}
        Beta schedule for the diffusion process (default: linear).

Model Architecture:

    --embedding_size EMBEDDING_SIZE
        Size of the input embedding (default: 128).
    --hidden_size HIDDEN_SIZE
        Size of the hidden layers (default: 256).
    --hidden_layers HIDDEN_LAYERS
        Number of hidden layers in the model.
    --time_embedding {sinusoidal, learnable, linear, zero}
        Type of time embedding to use.
    --input_embedding {sinusoidal, learnable, linear, identity}
        Type of input embedding to use.

Saving & Logging:

    --save_images_step SAVE_IMAGES_STEP
        Number of steps between saving generated images (default: 100).
    --log_level {DEBUG, INFO, WARNING, ERROR, CRITICAL}
        Logging level for the experiment (default: INFO).

Example Usage
```bash
python main.py train_ddpm --experiment_name "ddpm_experiment_1" --epochs 100 --dataset moons \
                         --train_batch_size 128 --eval_batch_size 64 --learning_rate 0.0001 \
                         --num_timesteps 1000 --beta_schedule linear --embedding_size 256 \
                         --hidden_size 512 --hidden_layers 4 --time_embedding sinusoidal \
                         --input_embedding learnable --save_images_step 200 --log_level INFO

```

### HP Search

From the root folder:
```bash
chmod +x hp_search.sh
./hp_search.sh
```
run the VAE model with different hyperparameters.