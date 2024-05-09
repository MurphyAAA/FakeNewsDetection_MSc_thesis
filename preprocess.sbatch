#!/bin/bash
#SBATCH --job-name=filter_images
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s300570@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --output=./result/filter_images_%j.log
#SBATCH --gres=gpu:1
source ~/.bashrc
conda activate NLP_project

python preprocessing/check_image.py
