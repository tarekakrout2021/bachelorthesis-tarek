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