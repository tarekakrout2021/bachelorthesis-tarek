#!/bin/bash
#SBATCH --job-name=gpu_job
#SBATCH --partition=gpu-2h
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv python_container.sif python3 src/main.py  --model "bitnet_synthetic" --epoch 10 --training_data "anisotropic" --learning_rate 0.001
