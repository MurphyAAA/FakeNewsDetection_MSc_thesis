# -*- coding: utf-8 -*-
"""
@Time ： 2024/6/27 15:08
@Auth ： Murphy
@File ：bert_vit_model.py
@IDE ：PyCharm
"""
import pdb

import torch
import torch.nn as nn
from transformers import BertModel, ViTModel, BertTokenizer, ViTImageProcessor, AutoModel, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoImageProcessor, AutoModelForImageClassification

class VisualSentimentModel(torch.nn.Module):
    def __init__(self):
        super(VisualSentimentModel, self).__init__()
        self.visual_sentiment = AutoModelForImageClassification.from_pretrained("kittendev/visual_emotional_analysis")
        self.category_encoder = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.BatchNorm1d(768),
            torch.nn.ReLU(),

            torch.nn.Linear(768, 768),
            torch.nn.BatchNorm1d(768),
            torch.nn.ReLU(),

            torch.nn.Linear(768, 768),
            torch.nn.BatchNorm1d(768),
            torch.nn.ReLU()
        )
        self.classifier = torch.nn.Linear(768, 3) # fc

    def forward(self, pixel_values):
        output = self.visual_sentiment(pixel_values=pixel_values, output_hidden_states=True)
        last_hidden_states = output['hidden_states'][-1]
        vis_emo_embeds = last_hidden_states[:, 0, :]
        vis_emo_embeds = self.category_encoder(vis_emo_embeds)
        x = self.classifier(vis_emo_embeds)
        return x,vis_emo_embeds

class MultimodalFusionBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultimodalFusionBlock, self).__init__()
        # Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, text_embeds, image_embeds):
        # Self-Attention: Intra-text dependencies
        # text_self, _ = self.self_attention(text_embeds, text_embeds, text_embeds)
        # text_embeds = self.self_attn_layer_norm(text_embeds + text_self)  # 残差连接 归一化
        # Cross-Attention: Text as Query, Image as Key/Value
        text_with_image, _ = self.cross_attention(text_embeds, image_embeds, image_embeds)
        fused_embeds = self.cross_attn_layer_norm(text_embeds + text_with_image)

        # Feed Forward Network
        fused_embeds = self.feed_forward(fused_embeds)
        # fused_embeds = self.ffn_layer_norm(fused_embeds + text_ffn)

        return fused_embeds
        
class Bert_VitClass(torch.nn.Module):
    def __init__(self, opt):
        super(Bert_VitClass, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')  # embedding
        self.img_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',num_labels=int(opt['label_type'][0]))
        self.text_sentiment= AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        # self.visual_sentiment= AutoModelForImageClassification.from_pretrained("kittendev/visual_emotional_analysis")
        self.visual_sentiment = VisualSentimentModel()
        self.intent_detector = AutoModelForSequenceClassification.from_pretrained("bespin-global/klue-roberta-small-3i4k-intent-classification" )
        self.multimodal_block = MultimodalFusionBlock(embed_dim=768, num_heads=opt['num_heads'])  # text-embedding.shape[1]
        self.processors = {
            'bert': BertTokenizer.from_pretrained('bert-base-uncased'),
            'vit': ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'),
            'text_sentiment': AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest"),
            'visual_sentiment': AutoImageProcessor.from_pretrained("kittendev/visual_emotional_analysis"),
            'intent_detector': AutoTokenizer.from_pretrained("bespin-global/klue-roberta-small-3i4k-intent-classification" )
        }
        fileName = f'{opt["output_path"]}/checkpoint_SA_model.pth'
        checkpoint = torch.load(fileName)
        self.visual_sentiment.load_state_dict(checkpoint['model'])
        # print(self.visual_sentiment)

        # self.dropout = torch.nn.Dropout(0.3)
        self.emotion_layer = torch.nn.Linear(768, 384)

        if opt["label_type"] == "2_way":
            self.category_classifier = torch.nn.Linear(768, 2)  # Bert base 的H是768
        elif opt["label_type"] == "3_way":
            self.category_classifier = torch.nn.Linear(768, 3)  # Bert base 的H是768
        else:  # 6_way
            self.category_classifier = torch.nn.Linear(768, 6)  # Bert base 的H是768

    def forward(self, ids, mask, token_type_ids, pixel_values, pixel_values_emo, emo_ids, emo_mask, intent_ids, intent_mask, labels):
        _, text_embeds = self.text_encoder(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False) # 16*768
        _, img_embeds = self.img_encoder(pixel_values=pixel_values, return_dict=False) # 16*768
        # _, emo_embeds = self.sentiment_model1(input_ids=emo_ids, attention_mask=emo_mask, return_dict=False)
        txt_emo_output = self.text_sentiment(input_ids=emo_ids, attention_mask=emo_mask, output_hidden_states=True) # 13* 16*100*768
        _, vis_emo_embeds = self.visual_sentiment(pixel_values=pixel_values_emo)  # 之后试一下用pixel_values
        # vis_emo_output = self.visual_sentiment(pixel_values=pixel_values, output_hidden_states=True) # 之后试一下用pixel_values
        txt_intent_output = self.intent_detector(input_ids=intent_ids, attention_mask=intent_mask, output_hidden_states=True)
        hidden_states = txt_emo_output['hidden_states']  # 16*100*768
        last_hidden_states = hidden_states[-1]
        txt_emo_embeds = last_hidden_states[:, 0, :] # cls token

        hidden_states = txt_intent_output['hidden_states']  # 16*100*768
        last_hidden_states = hidden_states[-1]
        txt_intent_embeds = last_hidden_states[:, 0, :]  # cls token

        # 计算 文字图片embedding的cross attention
        # combined_output = torch.cat((text_embeds, img_embeds), dim=1) # 试一下不直接将emotion的embedding拼起来，直接让模型返回text_embed和emo_embed，计算L2loss，让原本的模型得到的text_embed能更好的提取情绪信息 (embedding distillation)
        fused_embed = self.multimodal_block(text_embeds, img_embeds)
        output = self.category_classifier(fused_embed)
        return output, text_embeds, img_embeds, txt_emo_embeds, vis_emo_embeds, txt_intent_embeds

