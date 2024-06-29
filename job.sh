#!/bin/bash
#SBATCH --job-name=mnist_job
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

apptainer run --nv python_container.sif bash -c "export PYTHONPATH=\$PYTHONPATH:`pwd`
cd src
python3 main.py --model 'baseline_synthetic' \
                --epoch 20 \
                --training_data 'spiral' \
                --learning_rate 0.001 \
                --encoder_layers 512 256 \
                --decoder_layers 256 512 \
                --latent_dim 20 \
                --id job-$SLURM_JOB_ID"
