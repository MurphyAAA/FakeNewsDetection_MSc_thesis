# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/5 17:45
@Auth ： Murphy
@File ：bert_exper.py
@IDE ：PyCharm
"""
import torch
from models.bert_model import BertClass
import time
import pdb

class BertExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        self.model = BertClass()
        # model() 调用__call__()
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # define loss function and optimizer
        self.loss_fun = torch.nn.CrossEntropyLoss()
        self.loss_fun2 = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['lr'])

    def set_dataloader(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    # return training time
    def train(self, epoch):
        tot_loss = 0
        print_loss = 0
        epoch_time = 0
        self.model.train()
        start_time = time.time()
        for idx, databatch in enumerate(self.train_loader):  # batch_size=32, 共564000个训练样本，17625个batch，循环17625次
            # unpack from dataloader
            ids = databatch["ids"].to(self.device, dtype=torch.long)
            mask = databatch["mask"].to(self.device, dtype=torch.long)
            token_type_ids = databatch["token_type_ids"].to(self.device, dtype=torch.long)
            label = databatch["label"].to(self.device, dtype=torch.long)
            # predict
            embedding = self.model(ids, mask, token_type_ids) # embedding

            self.optimizer.zero_grad()
            # pdb.set_trace()
            loss = self.loss_fun(embedding, label)
            # for visualization
            tot_loss += loss.item()
            print_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if idx % self.opt["print_every"] == 0:
                end_time = time.time()
                loader_time = (end_time - start_time)
                epoch_time += loader_time
                start_time = time.time()
                print(
                    f"Epoch: {epoch}, batch: {len(self.train_loader) + 1}/{idx + 1}, avg_loss: {tot_loss / (idx + 1)}, loss_per_{self.opt['print_every']}: {print_loss / self.opt['print_every']}, time:{loader_time:.2f}s")  # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                print_loss = 0

            # if idx % opt["val_every"] == 0:
            #     outputs, labels = validate()
            #     acc = metrics.accuracy_score(labels, outputs)
            #     print(f'**acc** : 【{acc*10000//1/100}%】')
            #     model.train()
        return epoch_time

    # return predict result and real label
    def validate(self):
        self.model.eval()
        fin_label = []
        fin_output = []
        with torch.no_grad():
            for _, databatch in enumerate(self.val_loader):
                ids = databatch["ids"].to(self.device, dtype=torch.long)
                mask = databatch["mask"].to(self.device, dtype=torch.long)
                token_type_ids = databatch["token_type_ids"].to(self.device, dtype=torch.long)
                label = databatch["label"]

                logits = self.model(ids, mask, token_type_ids)
                pred = torch.argmax(logits, dim=-1)
                fin_label.extend(label.cpu().detach().tolist())
                fin_output.extend(pred.cpu().detach().tolist())
                # pdb.set_trace()
        return fin_output, fin_label
