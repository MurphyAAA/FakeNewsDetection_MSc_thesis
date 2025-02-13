# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/15 9:18
@Auth ： Murphy
@File ：clip_model.py
@IDE ：PyCharm
"""
import pdb

import torch
import torch.nn as nn
import transformers
from transformers import CLIPModel, CLIPProcessor


class ClipClass(torch.nn.Module):
    def __init__(self, opt):
        super(ClipClass, self).__init__()
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        if opt["model"] == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:  # clip_large
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # 冻结CLIP模型的参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 获取CLIP的输出维度
        # hidden_size = self.model.config.hidden_size

        # 自定义分类头（可根据需求修改结构）

        self.l3 = torch.nn.Dropout(0.2)
        if opt["label_type"] == "2_way":
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 2)
            )
            # self.l4 = torch.nn.Linear(1024, 2) # 只有text。没有image 所以临时改成512
        elif opt["label_type"] == "3_way":
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 3)
            )
            # self.l4 = torch.nn.Linear(1024, 3)
        else:  # 6_way
            self.classifier = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 6)
            )
            # self.l4 = torch.nn.Linear(1024, 6)

    # def __call__(self, *args, **kwargs):
    #     print("call Bert Class")
    def forward(self, ids=None, mask=None, pixel_values=None):
        # text+image
        output_1 = self.model(input_ids=ids, attention_mask=mask, pixel_values=pixel_values, output_hidden_states=True)  # 本任务更关注text和img的关系，而不是根据一个分类另一个
        text_embeds, img_embeds = output_1.text_embeds, output_1.image_embeds
        # clip text only
        # text_embeds = self.model.get_text_features(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        # text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        #clip image only
        # img_embeds = self.model.get_image_features(pixel_values=pixel_values, output_hidden_states=True)
        # print(f"------------------------------------------image embed:{img_embeds}, text embed:{text_embeds}") # 检查loss、变成nan的时候embedding是不是过大
        # output_2_img = self.l2(img_embeds)
        # output_2_text = self.l2(text_embeds)
        # combined_output = torch.cat((output_2_text, output_2_img), dim=1)
        combined_output = torch.cat((text_embeds, img_embeds), dim=1)  # ********** 组合方式也可以调整
        # output_3 = self.l3(combined_output)
        # output_3 = self.l3(img_embeds) # 先只看text embedding 为啥是nan了
        output = self.classifier(combined_output)
        return output
