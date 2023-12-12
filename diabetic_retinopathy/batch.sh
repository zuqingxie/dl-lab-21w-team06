#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=Plan_bbbb
#SBATCH --output=job-train-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 
# Activate everything you need
module load cuda/11.2


# Training with different models: vgg_like, CNN_team06, resnet_simple
# Or transfer learning model: mobilenet, inceptionV3,  inception_resnet_v2,  xception
python3 main.py -train=True -model_name="vgg_like"

## Run ensemble learning
#python3 ensemble.py

## evaluate the given latest Checkponit without training
#python3 main.py -train=False -eval_folder_name="run_xxxx-xx-xxxxx-xx-xx-xxxxxx"
