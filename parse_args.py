# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/24 9:58
@Auth ： Murphy
@File ：parse_args.py.py
@IDE ：PyCharm
"""
import argparse
import torch
def parse_arguments():
    parse = argparse.ArgumentParser()

    #hyperparams
    parse.add_argument('--lr', type=float, default=5e-05, help='Learning rate.')# 5e-5, 4e-5, 3e-5
    parse.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.') # 3
    parse.add_argument('--batch_size', type=int, default=32)

    parse.add_argument('--num_workers', type=int, default=1)
    parse.add_argument('--max_len', type=int, default=100) # 文本的最大长度
    parse.add_argument('--print_every', type=int, default=50)
    parse.add_argument('--val_every', type=int, default=150)

    parse.add_argument('--output_path', type=str, default='result',
                        help='Where to create the output directory containing logs and weights.')
    parse.add_argument('--data_path', type=str, default='data/Fakeddit', help='Locate the Fakeddit dataset on disk.')
    # parse.add_argument('--data_path', type=str, default='./FakeNewsDetection_MSc_thesis/data/Fakeddit',
    #                    help='Locate the Fakeddit dataset on disk.')

    parse.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU')
    parse.add_argument('--model', type=str, default="albef", choices=['bert', 'clip', 'clip_large', 'vit', 'vit_large', 'bert_vit', 'albef'])
    parse.add_argument('--label_type', type=str, default="2_way", choices=['2_way', '3_way', '6_way'])

    #Build options dict
    opt = vars(parse.parse_args())

    if not opt['cpu']:
        assert  torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

    return opt