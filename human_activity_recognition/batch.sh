#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=Plan_bbbb
#SBATCH --output=job-train-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 
# Activate everything you need
module load cuda/11.2


# Training the rnn model, the type of the model pls choose it in the config.gin
python3 main.py -train=True -device_name='GPU-Server' -model_name='rnn' -ensemble_learning=False

## Run ensemble learning
#python3 main.py -train=False -device_name='GPU-Server' -ensemble_learning=True

## Training the rnn_mix model
#python3 main.py -train=True -device_name='GPU-Server' -model_name='rnn_mix' -ensemble_learning=False

## Evaluate a specific Checkpoint
#python3 main.py -train=False -device_name='GPU-Server' -ensemble_learning=False -eval_folder_name="run2_2022-02-07T22-20-23-685508" -index_ckpt=2
