# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 15:08
@Auth ： Murphy
@File ：bert_vit_model.py
@IDE ：PyCharm
"""
import pdb

import torch
from transformers import BertModel, ViTModel, BertTokenizer, ViTImageProcessor, AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class Bert_VitClass(torch.nn.Module):
    def __init__(self, opt):
        super(Bert_VitClass, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')  # embedding
        self.img_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',
                                                               num_labels=int(opt['label_type'][0]))
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        # self.sentiment_model1 = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # self.dropout = torch.nn.Dropout(0.3)
        self.emotion_layer = torch.nn.Linear(768, 384)

        if opt["label_type"] == "2_way":
            self.category_classifier = torch.nn.Linear(768+768, 2)  # Bert base 的H是768
        elif opt["label_type"] == "3_way":
            self.category_classifier = torch.nn.Linear(768+768, 3)  # Bert base 的H是768
        else:  # 6_way
            self.category_classifier = torch.nn.Linear(768+768, 6)  # Bert base 的H是768

    def forward(self, ids, mask, token_type_ids, pixel_values, emo_ids, emo_mask, labels):
        _, text_embeds = self.text_encoder(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False) # 16*768
        _, img_embeds = self.img_encoder(pixel_values=pixel_values, return_dict=False) # 16*768
        # _, emo_embeds = self.sentiment_model1(input_ids=emo_ids, attention_mask=emo_mask, return_dict=False)
        emo_output = self.sentiment_model(input_ids=emo_ids, attention_mask=emo_mask, output_hidden_states=True) # 13* 16*100*768
        hidden_states = emo_output['hidden_states']  # 16*100*768
        last_hidden_states = hidden_states[-1]
        # emo_embeds = last_hidden_states.mean(dim=1) # pooling 16*768 .max(dim=1).values
        emo_embeds = last_hidden_states[:, 0, :] # cls token
        # emo_embeds = self.emotion_layer(emo_embeds)

        combined_output = torch.cat((text_embeds, img_embeds), dim=1) # 试一下不直接将emotion的embedding拼起来，直接让模型返回text_embed和emo_embed，计算L2loss，让原本的模型得到的text_embed能更好的提取情绪信息 (embedding distillation)
        output = self.category_classifier(combined_output)
        return output, text_embeds, emo_embeds #last_hidden_states[:, 0, :]

