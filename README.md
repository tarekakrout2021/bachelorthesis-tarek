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
apptainer build --fakeroot main.sif main.def
```

### Run the code
From the src folder (example):
```bash
python3 main.py --model "bitnet_vae" --training_data "anisotropic" --epochs 10 --batch_size 32 --learning_rate 0.001 
```