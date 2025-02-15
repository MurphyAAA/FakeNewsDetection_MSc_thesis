# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/24 21:19
@Auth ： Murphy
@File ：bert_model.py
@IDE ：PyCharm
"""
import torch
import transformers
from transformers import BertTokenizer


class BertClass(torch.nn.Module):
    def __init__(self,opt):
        super(BertClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased') # embedding
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # self.l2 = torch.nn.Dropout(0.3)
        if opt["label_type"] == "2_way":
            self.l3 = torch.nn.Linear(768, 2) # Bert base 的H是768
        elif opt["label_type"] == "3_way":
            self.l3 = torch.nn.Linear(768, 3)  # Bert base 的H是768
        else:  # 6_way
            self.l3 = torch.nn.Linear(768, 6)  # Bert base 的H是768

    # def __call__(self, *args, **kwargs):
    #     print("call Bert Class")
    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        # output_2 = self.l2(output_1)
        output = self.l3(output_1)
        return output

