#!/bin/bash
#SBATCH --job-name=fake_news_detection
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s300570@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=06:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --output=./result/clip_result_2_way_%j.log
# --output=fake_news_detection_%j.log
#SBATCH --gres=gpu:3
source ~/.bashrc
conda activate NLP_project

python main.py --batch_size=16 --num_epochs=1
