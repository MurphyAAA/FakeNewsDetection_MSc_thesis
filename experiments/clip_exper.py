# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/5 20:22
@Auth ： Murphy
@File ：clip_exper.py
@IDE ：PyCharm
"""
import torch
import clip
import time
import pdb

class ClipExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        self.model, self.preprocess = clip.load("ViT-B/32", device='cpu') # load it first to CPU to ensure you're using fp32 precision
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.mse_loss = torch.nn.MSELoss()
        self.ent_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['lr'], betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)#Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    def set_dataloader(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def convert_models_to_fp32(self):
        for p in self.model.parameters():
            p.data = p.data.float()
            p.grad.data = p.grad.data.float()
    def train(self, epoch):
        tot_loss = 0
        print_loss = 0
        epoch_time=0
        self.model.train()
        start_time = time.time()
        for idx, databatch in enumerate(self.train_loader):
            # img, description,label
            img = databatch["img"].to(self.device)
            text_token = databatch["text_token"].to(self.device)
            # label =
            logits_per_image, logits_per_text = self.model(img, text_token)
            ground_truth = torch.arange(len(img), dtype=torch.long, device=self.device)

            self.optimizer.zero_grad()
            loss = (self.ent_loss(logits_per_image, ground_truth) + self.ent_loss(logits_per_text, ground_truth)) / 2
            tot_loss += loss.item()
            print_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            if self.device == "cpu":
                self.optimizer.step()
            else:
                self.convert_models_to_fp32()
                self.optimizer.step()
                clip.model.convert_weights(self.model)  # mixed precision training. Convert applicable model parameters to fp16

            if idx % self.opt["print_every"] == 0:
                end_time = time.time()
                loader_time = (end_time - start_time)
                epoch_time += loader_time
                start_time = time.time()
                print(
                    f"Epoch: {epoch}, batch: {len(self.train_loader) + 1}/{idx + 1}, avg_loss: {tot_loss / (idx + 1)}, loss_per_{self.opt['print_every']}: {print_loss / self.opt['print_every']}, time:{loader_time:.2f}s")  # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                print_loss = 0
        return epoch_time


    # def validation(self):
    #     # todo
