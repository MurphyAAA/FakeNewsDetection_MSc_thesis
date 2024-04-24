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

    parse.add_argument('--lr', type=float, default=1e-05, help='Learning rate.')
    parse.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs.')
    parse.add_argument('--batch_size', type=int, default=32)
    parse.add_argument('--num_workers', type=int, default=1)
    parse.add_argument('--max_len', type=int, default=200) # 文本的最大长度

    parse.add_argument('--data_path', type=str, default='data/Fakeddit', help='Locate the Fakeddit dataset on disk.')


    parse.add_argument('--cpu', action='store_true', help='If set, the experiment will run on the CPU')

    #Build options dict
    opt = vars(parse.parse_args())

    if not opt['cpu']:
        assert  torch.cuda.is_available(), 'You need a CUDA capable device in order to run this experiment. See `--cpu` flag.'

    return opt