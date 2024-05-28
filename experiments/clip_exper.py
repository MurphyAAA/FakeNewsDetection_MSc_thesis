# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/5 20:22
@Auth ： Murphy
@File ：clip_exper.py
@IDE ：PyCharm
"""
import numpy
import torch
from transformers import CLIPProcessor
from models.clip_model import ClipClass
import time
from torch.cuda.amp import autocast, GradScaler
import pdb

# torch.autograd.set_detect_anomaly(True)

def check_gradients(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class ClipExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        # self.model, self.preprocess = clip.load("ViT-B/32", device='cpu') # load it first to CPU to ensure you're using fp32 precision
        # self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model = ClipClass(opt)
        if opt["model"] == "clip":
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else: # clip_large
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.mse_loss = torch.nn.MSELoss()
        self.ent_loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=opt['lr'], betas=(0.9, 0.98), eps=1e-6,
                                          weight_decay=0.2)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

        # self.scaler = GradScaler()
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")

    def set_dataloader(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    # def convert_models_to_fp32(self):
    #     for p in self.model.parameters():
    #         p.data = p.data.float()
    #         p.grad.data = p.grad.data.float()

    def save_clip_checkpoint(self, path, epoch):
        checkpoint = {
            'end_epoch': epoch,
            # 'tot_loss': tot_loss,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            # 'scaler': self.scaler.state_dict()
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch}")

    def load_clip_checkpoint(self, path):
        checkpoint = torch.load(path)
        epoch = checkpoint['end_epoch']+1
        # tot_loss = checkpoint['tot_loss']
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.scaler.load_state_dict(checkpoint['scaler'])
        print(f"Checkpoint loaded. Resuming training from epoch {epoch}")
        return epoch

    def freeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def train(self, epoch):
        tot_loss = 0
        print_loss = 0
        epoch_time = 0
        self.model.train()
        start_time = time.time()
        for idx, databatch in enumerate(self.train_loader):
            ids = databatch["ids"].to(self.device, dtype=torch.long)
            mask = databatch["mask"].to(self.device, dtype=torch.long)
            pixel_values = databatch["pixel_values"].to(self.device, dtype=torch.float)
            label = databatch["label"].to(self.device, dtype=torch.long)
        # with autocast():  # mixed precision training. Convert applicable model parameters to fp16  **********先不加混精度试一下
                # logits_per_image, logits_per_text = self.model(**{"input_ids":ids, "attention_mask":mask, "pixel_values":pixel_values})
                # output = self.model(input_ids=ids, pixel_values=pixel_values, attention_mask=mask, return_loss=True)
            output = self.model(ids, mask, pixel_values)
            self.optimizer.zero_grad()
            loss = self.ent_loss(output, label)
            tot_loss += loss.item()
            print_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scaler.scale(loss).backward()  # 对缩放后的损失进行反向传播
            if torch.isnan(loss):
                print(f"loss is nan: [{loss.item()}, {output}, {label}, {idx}]")
                grad_norm = check_gradients(self.model) # 检查loss变成nan的时候是否梯度爆炸
                print(f"gradient: {grad_norm}") # 梯度消失了???

            # # 梯度裁剪 防止梯度过大loss变成nan
            # self.scaler.unscale_(self.optimizer)  # 在裁剪之前，确保梯度是未缩放的
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # self.scaler.step(self.optimizer)  # 来更新模型参数。
            # self.scaler.update()  # 来更新模型参数。
            if idx % self.opt["print_every"] == 0:
                end_time = time.time()
                loader_time = (end_time - start_time)
                epoch_time += loader_time
                start_time = time.time()
                if numpy.isnan(tot_loss / (idx + 1)):
                    print(f"avg loss is nan: [ {tot_loss} / {(idx + 1)} ]")

                print(
                    f"Epoch: {epoch}, batch: {len(self.train_loader) + 1}/{idx + 1}, avg_loss: {tot_loss / (idx + 1)}, loss_per_{self.opt['print_every']}: {print_loss / self.opt['print_every']}, time:{loader_time:.2f}s")  # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss

                print_loss = 0
        # self.scheduler.step(tot_loss)
        return epoch_time

    def validation(self):
        self.model.eval()
        tot_loss = 0
        fin_label = []
        fin_output = []
        with torch.no_grad():
            for _, databatch in enumerate(self.val_loader):
                ids = databatch["ids"].to(self.device, dtype=torch.long)
                mask = databatch["mask"].to(self.device, dtype=torch.long)
                pixel_values = databatch["pixel_values"].to(self.device, dtype=torch.float)
                label = databatch["label"].to(self.device, dtype=torch.long)

                embedding = self.model(ids, mask, pixel_values)#  , pixel_values
                loss = self.ent_loss(embedding, label)
                pred = torch.argmax(embedding, dim=-1)
                tot_loss += loss.item()
                if _ % self.opt["print_every"] == 0:
                    print(
                        f" avg_loss: {tot_loss / (_ + 1)}")  # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                fin_label.extend(label.cpu().detach().tolist())
                fin_output.extend(pred.cpu().detach().tolist())
        return fin_output, fin_label
