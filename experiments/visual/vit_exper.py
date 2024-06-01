# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/30 14:04
@Auth ： Murphy
@File ：vit_exper.py
@IDE ：PyCharm
"""
import pdb
import time

import torch
from models.vit_model import VitClass
from transformers import ViTImageProcessor


class VitExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        self.model = VitClass(opt)
        # model() 调用__call__()
        self.model.to(self.device)
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # define loss function and optimizer
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['lr'])

    def set_dataloader(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def save_checkpoint(self, path, epoch):
        checkpoint = {
            'end_epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        epoch = checkpoint['end_epoch']+1
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}")
        return epoch

    # def train(self, epoch):
        #  todo
        # tot_loss = 0
        # print_loss = 0
        # epoch_time = 0
        # self.model.train()
        # start_time = time.time()
        # for idx, databatch in enumerate(self.train_loader):
        #     pixel_values = databatch["pixel_values"].to(self.device, dtype=torch.float)
        #     labels = databatch["labels"].to(self.device, dtype=torch.long)
        #     pdb.set_trace()
    def collate_fn(self, batch):
        # print(torch.stack([x['pixel_values'] for x in batch]).shape) # 看一下是不是构建batch有问题，导致和attention head对不上，12*16

        return {
            # 'pixel_values': torch.squeeze(torch.stack([x['pixel_values'] for x in batch]), dim=1),
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }