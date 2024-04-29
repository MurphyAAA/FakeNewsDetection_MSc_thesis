#!/bin/bash
#SBATCH --job-name=fake_news_detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s300570@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --output=fake_news_detection_%j.log
#SBATCH --gres=gpu:1

conda activate NLP_project

python main.py
