# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 15:08
@Auth ： Murphy
@File ：bert_vit_model.py
@IDE ：PyCharm
"""

import torch
from transformers import BertModel, ViTForImageClassification

class Bert_VitClass(torch.nn.Module):
    def __init__(self, opt):
        super(Bert_VitClass, self).__init__()
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')  # embedding
        self.vitmodel = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                               num_labels=int(opt['label_type'][0]))
        self.dropout = torch.nn.Dropout(0.3)

        if opt["label_type"] == "2_way":
            self.fc = torch.nn.Linear(768, 2)  # Bert base 的H是768
        elif opt["label_type"] == "3_way":
            self.fc = torch.nn.Linear(768, 3)  # Bert base 的H是768
        else:  # 6_way
            self.fc = torch.nn.Linear(768, 6)  # Bert base 的H是768

    def forward(self, ids, mask, token_type_ids, pixel_values, labels):
        _, text_embeds = self.bertmodel(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_vit = self.vitmodel(pixel_values=pixel_values, labels=labels)
        print(f"output_bert:{text_embeds}")
        print(f"output_vit:{output_vit}") #####
        # output_2 = self.dropout(output_1)
        # output = self.fc(output_2)
        return 1

