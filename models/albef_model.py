# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/3 18:32
@Auth ： Murphy
@File ：albef_model.py
@IDE ：PyCharm
"""
import pdb

import torch
from transformers import BertModel, ViTModel
from lavis.models import load_model_and_preprocess
class AlbefClass(torch.nn.Module):
    def __init__(self, opt):
        super(AlbefClass, self).__init__()
        self.device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
        # name=albef_feature_extractor
        self.albefmodel, self.img_processor, self.text_processor = load_model_and_preprocess("albef_feature_extractor", model_type="base", device=self.device)
        # print(self.img_processor)
        # print(self.text_processor)
        self.dropout = torch.nn.Dropout(0.3)
        if opt["label_type"] == "2_way":
            self.fc = torch.nn.Linear(256+256, 2)  # project embedding 256
        elif opt["label_type"] == "3_way":
            self.fc = torch.nn.Linear(256+256, 3)
        else:  # 6_way
            self.fc = torch.nn.Linear(256+256, 6)

    def forward(self, sample):
        text_feature = self.albefmodel.extract_features(sample, mode="text")
        text_embeds = text_feature.text_embeds_proj[:, 0, :]

        img_feature = self.albefmodel.extract_features(sample, mode="image")
        img_embeds = img_feature.image_embeds_proj[:, 0, :]
        # print(text_embeds.shape)
        # pdb.set_trace()

        combined_output = torch.cat((text_embeds, img_embeds), dim=1)
        output = self.dropout(combined_output)
        output = self.fc(output)
        return output


