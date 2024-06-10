#!/bin/bash
#SBATCH --job-name=mnist_job
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv python_container.sif bash -c "export PYTHONPATH=\$PYTHONPATH:`pwd` && cd src && python3 main.py --model 'bitnet_mnist' --epoch 2 --training_data 'mnist' --learning_rate 0.001"
