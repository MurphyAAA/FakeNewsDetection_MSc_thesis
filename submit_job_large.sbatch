#!/bin/bash
#SBATCH --job-name=clip_large
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s300570@studenti.polito.it
#SBATCH --partition=cuda
#SBATCH --time=06:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --output=./result/clip_large_result_2_way_%j.log
# --output=fake_news_detection_%j.log
#SBATCH --gres=gpu:2
source ~/.bashrc
conda activate NLP_project

python main.py --batch_size=8 --label_type="2_way" --model="clip_large"
