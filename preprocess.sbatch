#!/bin/bash
#SBATCH --job-name=filter images
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s300570@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --output=./result/clip_result_2_way_%j.log
# --output=filter_image_%j.log
#SBATCH --gres=gpu:1
source ~/.bashrc
conda activate NLP_project

python ./preprocessing/check_image.py
