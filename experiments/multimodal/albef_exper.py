# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/3 13:13
@Auth ： Murphy
@File ：albef_exper.py
@IDE ：PyCharm
"""
import pdb

# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 14:47
@Auth ： Murphy
@File ：bert_vit_exper.py
@IDE ：PyCharm
"""
import torch
# from models.bert_vit_model import Bert_VitClass
import time
from models.albef_model import AlbefClass
from lavis.models import load_model_and_preprocess
class AlbefExperiment:
    def __init__(self, opt):

        self.opt = opt
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        self.model = AlbefClass(opt)
        self.model.to(self.device)
        self.text_processor = self.model.text_processor
        self.img_processor = self.model.img_processor

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.ent_loss = torch.nn.CrossEntropyLoss()
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
    def train(self, epoch):
        tot_loss = 0
        print_loss = 0
        epoch_time =0
        self.model.train()
        start_time = time.time()
        epoch_start = time.time()
        for idx, databatch in enumerate(self.train_loader):
            # print(databatch)
            image = databatch["image"].to(self.device)
            text_input = databatch["text_input"]
            labels = databatch["labels"].to(self.device, dtype=torch.long)
            # image_1 = self.img_processor["eval"](image).unsqueeze(0).to(self.device)
            # text_input_1 = self.text_processor["eval"](text_input)
            sample = {"image": image, "text_input": text_input}

            logits = self.model(sample)
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
                image = databatch["image"].to(self.device)
                text_input = databatch["text_input"]
                labels = databatch["labels"].to(self.device, dtype=torch.long)
                # image_1 = self.img_processor["eval"](image).unsqueeze(0).to(self.device)
                # text_input_1 = self.text_processor["eval"](text_input)
                sample = {"image": image, "text_input": text_input}

                logits = self.model(sample)
                pred = torch.argmax(logits, dim=-1)
                fin_label.extend(labels.cpu().detach().tolist())
                fin_output.extend(pred.cpu().detach().tolist())
                # pdb.set_trace()
        return fin_output, fin_label
