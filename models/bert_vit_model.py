# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 15:08
@Auth ： Murphy
@File ：bert_vit_model.py
@IDE ：PyCharm
"""
import pdb

import torch
from transformers import BertModel, ViTModel, BertTokenizer, ViTImageProcessor, DistilBertTokenizer, \
    DistilBertForSequenceClassification, AutoModel, AutoTokenizer


class Bert_VitClass(torch.nn.Module):
    def __init__(self, opt):
        super(Bert_VitClass, self).__init__()
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')  # embedding
        self.vitmodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',
                                                               num_labels=int(opt['label_type'][0]))
        self.sentiment_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # self.dropout = torch.nn.Dropout(0.3)
        self.emotion_layer = torch.nn.Linear(4, 128)
        self.bn = torch.nn.BatchNorm1d(128)
        self.relu = torch.nn.ReLU()
        if opt["label_type"] == "2_way":
            self.fc = torch.nn.Linear(768+128+768, 2)  # Bert base 的H是768
        elif opt["label_type"] == "3_way":
            self.fc = torch.nn.Linear(768+128+768, 3)  # Bert base 的H是768
        else:  # 6_way
            self.fc = torch.nn.Linear(768+128+768, 6)  # Bert base 的H是768

    def forward(self, ids, mask, token_type_ids, pixel_values, emotions, labels):
        _, text_embeds = self.bertmodel(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        _, img_embeds = self.vitmodel(pixel_values=pixel_values, return_dict=False)
        emotions = self.emotion_layer(emotions)
        emotions = self.bn(emotions)
        emotions = self.relu(emotions)
        combined_output = torch.cat((text_embeds, emotions, img_embeds), dim=1)
        # output = self.dropout(combined_output)
        output = self.fc(combined_output)
        return output

