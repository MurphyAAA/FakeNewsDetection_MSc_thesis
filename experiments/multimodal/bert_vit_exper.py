# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 14:47
@Auth ： Murphy
@File ：bert_vit_exper.py
@IDE ：PyCharm
"""
import pdb

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from models.bert_vit_model import Bert_VitClass
import time
from scipy.special import softmax
class Bert_VitExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.writer = SummaryWriter(opt['log_dir']+opt['model'])
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        self.model = Bert_VitClass(opt)
        self.tokenizer = self.model.tokenizer
        self.vit_processor = self.model.vit_processor
        self.sentiment_tokenizer = self.model.sentiment_tokenizer
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.ent_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['lr'])
        # self.optimizer = torch.optim.AdamW([
        #     {'params': self.model.bertmodel.parameters(), 'lr': 5e-5},  # 对BERT部分使用较小的学习率
        #     {'params': self.model.vitmodel.parameters(), 'lr': 5e-4},  # 对ViT部分使用稍大的学习率
        #     {'params': self.model.fc.parameters(), 'lr': 5e-3},  # 对全连接层使用更大的学习率
        # ])

    def set_weighted_loss(self, class_weight):
        class_weight = class_weight.to(self.device, dtype=torch.float)
        self.ent_loss = torch.nn.CrossEntropyLoss(weight=class_weight)
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
    def train(self, epoch, tot_loss):
        print(f"Start training at epoch {epoch}")
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
            emo_ids = databatch["emo_ids"].to(self.device, dtype=torch.long)
            emo_mask = databatch["emo_mask"].to(self.device, dtype=torch.long)
            # 现在返回的就是一个1，要看一下模型输出，找到embedding
            logits = self.model(ids, mask, token_type_ids, pixel_values, emo_ids, emo_mask, labels)

            self.optimizer.zero_grad()
            loss = self.ent_loss(logits, labels)
            self.writer.add_scalar(f"loss_{self.opt['label_type']}", loss.item(), epoch*len(self.train_loader) + idx)
            tot_loss += loss.item()
            print_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if idx % self.opt["print_every"] == 0:
                end_time = time.time()
                batch_time = (end_time - start_time)
                start_time = time.time()
                print(
                    f"Epoch: {epoch}, batch: {len(self.train_loader) + 1}/{idx + 1}, avg_loss: {tot_loss / (epoch*len(self.train_loader)+idx + 1)}, loss_per_{self.opt['print_every']}: {print_loss / self.opt['print_every']}, time:{batch_time:.2f}s")  # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                print_loss = 0
        epoch_end = time.time()
        epoch_time = epoch_end- epoch_start
        return epoch_time, tot_loss

    def validation(self):
        print("Validation Started")
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
                emo_ids = databatch["emo_ids"].to(self.device, dtype=torch.long)
                emo_mask = databatch["emo_mask"].to(self.device, dtype=torch.long)
                logits = self.model(ids, mask, token_type_ids, pixel_values, emo_ids, emo_mask, labels)

                pred = torch.argmax(logits, dim=-1)
                fin_label.extend(labels.cpu().detach().tolist())
                fin_output.extend(pred.cpu().detach().tolist())
                # pdb.set_trace()
        return fin_output, fin_label
