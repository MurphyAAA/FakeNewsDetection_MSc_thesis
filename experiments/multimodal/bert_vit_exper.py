# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 14:47
@Auth ： Murphy
@File ：bert_vit_exper.py
@IDE ：PyCharm
"""
import torch
from models.bert_vit_model import Bert_VitClass
import time

class Bert_VitExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        self.model = Bert_VitClass(opt)
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.ent_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['lr'])

    def set_dataloader(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train(self, epoch):
        tot_loss = 0
        print_loss = 0
        epoch_time =0
        self.model.train()
        start_time = time.time()
        epoch_start = time.time()
        for idx, databatch in enumerate(self.train_loader):
            ids = databatch["ids"].to(self.device, dtype=torch.long)
            mask = databatch["mask"].to(self.device, dtype=torch.long)
            token_type_ids = databatch["token_type_ids"].to(self.device, dtype=torch.long)
            labels = databatch["labels"].to(self.device, dtype=torch.long)
            pixel_values = databatch["pixel_values"].to(self.device, dtype=torch.float)
            # 现在返回的就是一个1，要看一下模型输出，找到embedding
            logits = self.model(ids, mask, token_type_ids, pixel_values, labels)

            self.optimizer.zero_grad()
            loss = self.ent_loss(logits, labels)
            tot_loss += loss.item()
            print_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if idx % self.opt["print_every"] == 0:
                end_time = time.time()
                loader_time = (end_time - start_time)
                start_time = time.time()
                print(
                    f"Epoch: {epoch}, batch: {len(self.train_loader) + 1}/{idx + 1}, avg_loss: {tot_loss / (idx + 1)}, loss_per_{self.opt['print_every']}: {print_loss / self.opt['print_every']}, time:{loader_time:.2f}s")  # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                print_loss = 0
        epoch_end = time.time()
        epoch_time = epoch_end- epoch_start
        return epoch_time

    def validation(self):
        self.model.eval()
        fin_label = []
        fin_output = []
        with torch.no_grad():
            for _, databatch in enumerate(self.val_loader):
                ids = databatch["ids"].to(self.device, dtype=torch.long)
                mask = databatch["mask"].to(self.device, dtype=torch.long)
                token_type_ids = databatch["token_type_ids"].to(self.device, dtype=torch.long)
                labels = databatch["labels"].to(self.device, dtype=torch.long)
                pixel_values = databatch["pixel_values"].to(self.device, dtype=torch.float)

                logits = self.model(ids, mask, token_type_ids, pixel_values, labels)

                pred = torch.argmax(logits, dim=-1)
                fin_label.extend(labels.cpu().detach().tolist())
                fin_output.extend(pred.cpu().detach().tolist())
                # pdb.set_trace()
        return fin_output, fin_label
