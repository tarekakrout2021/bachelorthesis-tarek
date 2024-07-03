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

From the src folder (example):
```bash
python3 main.py --model "bitnet_synthetic" --training_data "anisotropic" --epochs 10 --batch_size 32 --learning_rate 0.001 
```

Models are named as `{model}_{dataset}` where dataset is either `synthetic` or `mnist` and  model is either `baseline` or `bitnet`.
Training data can either be `anisotropic`, `normal`, `spiral` or `mnist`.


From the root folder:
```bash
chmod +x hp_search.sh
./hp_search.sh
```
run the model with different hyperparameters.