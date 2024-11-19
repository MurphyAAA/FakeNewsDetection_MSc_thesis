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
    DistilBertForSequenceClassification, AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class Bert_VitClass(torch.nn.Module):
    def __init__(self, opt):
        super(Bert_VitClass, self).__init__()
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')  # embedding
        self.vitmodel = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',
                                                               num_labels=int(opt['label_type'][0]))
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.sentiment_model1 = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # self.dropout = torch.nn.Dropout(0.3)
        self.emotion_layer = torch.nn.Linear(768, 384)
        self.bn = torch.nn.BatchNorm1d(384)
        self.relu = torch.nn.ReLU()
        if opt["label_type"] == "2_way":
            self.fc = torch.nn.Linear(768+384+768, 2)  # Bert base 的H是768
        elif opt["label_type"] == "3_way":
            self.fc = torch.nn.Linear(768+384+768, 3)  # Bert base 的H是768
        else:  # 6_way
            self.fc = torch.nn.Linear(768+384+768, 6)  # Bert base 的H是768

    def forward(self, ids, mask, token_type_ids, pixel_values, emo_ids, emo_mask, labels):
        _, text_embeds = self.bertmodel(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False) # 16*768
        _, img_embeds = self.vitmodel(pixel_values=pixel_values, return_dict=False) # 16*768
        # _, emo_embeds = self.sentiment_model1(input_ids=emo_ids, attention_mask=emo_mask, return_dict=False)
        emo_output = self.sentiment_model(input_ids=emo_ids, attention_mask=emo_mask, output_hidden_states=True) # 13* 16*100*768
        hidden_states = emo_output['hidden_states']  # 16*100*768
        last_hidden_states = hidden_states[-1]  # [:,0,:]
        emo_embeds = last_hidden_states.mean(dim=1) # pooling 16*768 .max(dim=1).values

        emo_embeds = self.emotion_layer(emo_embeds)
        emo_embeds = self.bn(emo_embeds)
        emo_embeds = self.relu(emo_embeds)
        combined_output = torch.cat((text_embeds, emo_embeds, img_embeds), dim=1)
        output = self.fc(combined_output)
        return output

