#!/bin/bash
#SBATCH --job-name=mnist_job
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j.out

# Define arrays of hyperparameter values
learning_rates=(0.001 0.0003)
epochs=(10 40) # 100
latent_dim=(2 5 8)
encoder_layers=("512 256" "512 256 128" )
decoder_layers=("256 512" "128 256 512")

# Iterate over each combination of hyperparameters
for lr in "${learning_rates[@]}"; do
  for epoch in "${epochs[@]}"; do
    for enc_layers in "${encoder_layers[@]}"; do
      for dec_layers in "${decoder_layers[@]}"; do
        for dim in "${latent_dim[@]}"; do
          sbatch --mail-user=tarek_akrout@tu-berlin.de <<EOT
#!/bin/bash
#SBATCH --job-name=mnist_job_lr${lr}_epoch${epoch}
#SBATCH --partition=cpu-2h
#SBATCH --ntasks-per-node=2
#SBATCH --output=logs/job-%j_lr${lr}_epoch${epoch}.out

apptainer run --nv python_container.sif bash -c "export PYTHONPATH=\$PYTHONPATH:\`pwd\` && cd src && python3 main.py --model 'bitnet_mnist' --epoch ${epoch} --training_data 'mnist' --learning_rate ${lr} --encoder_layers ${enc_layers} --decoder_layers ${dec_layers} --latent_dim 8"
EOT
        done
      done
    done
  done
done