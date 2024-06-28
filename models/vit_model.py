# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/30 16:24
@Auth ： Murphy
@File ：vit_model.py
@IDE ：PyCharm
"""
import pdb

import torch
from transformers import ViTForImageClassification

class VitClass(torch.nn.Module):
    def __init__(self,opt):
        super(VitClass, self).__init__()
        if opt["model"] == "vit":
            # self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=int(opt['label_type'][0]))
        else:  # vit_large
            # self.feature_extractor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
            self.model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=int(opt['label_type'][0]))

        # self.l2 = torch.nn.Dropout(0.3)
        # if opt["label_type"] == "2_way":
        #     self.l3 = torch.nn.Linear(512, 2)
        # elif opt["label_type"] == "3_way":
        #     self.l3 = torch.nn.Linear(512, 3)
        # else:  # 6_way
        #     self.l3 = torch.nn.Linear(512, 6)

    # def __call__(self, *args, **kwargs):
    #     print("call Bert Class")
    def forward(self,pixel_values, labels):
        output = self.model(pixel_values=pixel_values, labels=labels)
        # output_2 = self.l2(output_1)
        # output = self.l3(output_2)
        return output