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

class TextSentimentModel(nn.Module):
    def __init__(self):
        super(TextSentimentModel, self).__init__()
        self.text_sentiment = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
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
        self.classifier = torch.nn.Linear(768, 5) # fc
    def forward(self, input_ids, attention_mask):
        output = self.text_sentiment(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # print(output.keys())
        last_hidden_states = output['hidden_states'][-1]
        txt_emo_embeds = last_hidden_states[:, 0, :]
        txt_emo_embeds = self.category_encoder(txt_emo_embeds)
        x=self.classifier(txt_emo_embeds)
        return x, txt_emo_embeds

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

# Model
class IntentModel(nn.Module):
    def __init__(self):
        super(IntentModel, self).__init__()
        self.intent_model = AutoModel.from_pretrained("roberta-base")
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
        self.classifier = torch.nn.Linear(768, 150) # fc
    def forward(self, input_ids, attention_mask):
        output = self.intent_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # print(output.keys())
        last_hidden_states = output['hidden_states'][-1]
        intent_embeds = last_hidden_states[:, 0, :]
        intent_embeds = self.category_encoder(intent_embeds)
        x=self.classifier(intent_embeds)
        return x, intent_embeds


class MultimodalFusionBlock(torch.nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super(MultimodalFusionBlock, self).__init__()
        # # Self-Attention
        # self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        #
        # # Cross-Attention
        # self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        #
        # # Feed Forward
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        # )
        # self.ffn_layer_norm = nn.LayerNorm(embed_dim)

        # 跨模态注意力层
        self.cross_attn = nn.ModuleList([
            CrossModalAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(2)
        ])

        # 自适应门控
        self.gate = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.Sigmoid()
        )
    def forward(self, text_embeds, image_embeds):
        # 双向交叉注意力
        for attn_layer in self.cross_attn:
            text, img = attn_layer(text_embeds, image_embeds)

        # 特征聚合
        text_pool = text.mean(dim=1)  # [B, 768]
        img_pool = img.mean(dim=1)  # [B, 768]

        # 动态门控融合
        gate = self.gate(torch.cat([text_pool, img_pool], dim=1))
        fused = gate * text_pool + (1 - gate) * img_pool
        # fused = fused + self.ffn(fused)
        return fused


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        # 文本到图像注意力
        self.txt2img_attn = nn.MultiheadAttention(embed_dim, num_heads)
        # 图像到文本注意力
        self.img2txt_attn = nn.MultiheadAttention(embed_dim, num_heads)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)



    def forward(self, text, img):
        # 维度转换 (B, S, D) -> (S, B, D)
        text_in = text.permute(1, 0, 2)
        img_in = img.permute(1, 0, 2)

        # 文本关注图像
        text_attn_out, _ = self.txt2img_attn(
            query=text_in,
            key=img_in,
            value=img_in
        )
        text = self.norm1(text + text_attn_out.permute(1, 0, 2))

        # 图像关注文本
        img_attn_out, _ = self.img2txt_attn(
            query=img_in,
            key=text_in,
            value=text_in
        )
        img = self.norm2(img + img_attn_out.permute(1, 0, 2))

        text = self.norm3(text + self.ffn(text)) # 模态内的非线性变换
        img = self.norm4(img + self.ffn(img))
        return text, img

class Bert_VitClass(torch.nn.Module):
    def __init__(self, opt):
        super(Bert_VitClass, self).__init__()
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')  # embedding
        self.img_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k',num_labels=int(opt['label_type'][0]))
        # self.text_sentiment= AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.text_sentiment = TextSentimentModel()
        # self.visual_sentiment= AutoModelForImageClassification.from_pretrained("kittendev/visual_emotional_analysis")
        self.visual_sentiment = VisualSentimentModel()
        # self.intent_detector = AutoModelForSequenceClassification.from_pretrained("bespin-global/klue-roberta-small-3i4k-intent-classification" )
        self.intent_detector = IntentModel()
        self.multimodal_block = MultimodalFusionBlock(embed_dim=768, num_heads=opt['num_heads'])  # text-embedding.shape[1]
        self.processors = {
            'bert': BertTokenizer.from_pretrained('bert-base-uncased'),
            'vit': ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k'),
            'text_sentiment': AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest"),
            'visual_sentiment': AutoImageProcessor.from_pretrained("kittendev/visual_emotional_analysis"),
            'intent_detector': AutoTokenizer.from_pretrained("roberta-base" )
        }
        visual_sa_model = f'{opt["output_path"]}/checkpoint_visSA_model.pth'
        checkpoint = torch.load(visual_sa_model)
        self.visual_sentiment.load_state_dict(checkpoint['model'])

        text_sa_model = f'{opt["output_path"]}/checkpoint_textSA_model.pth'
        checkpoint = torch.load(text_sa_model)
        self.text_sentiment.load_state_dict(checkpoint['model'])

        text_intent_model = f'{opt["output_path"]}/checkpoint_intent_model.pth'
        checkpoint = torch.load(text_intent_model)
        self.intent_detector.load_state_dict(checkpoint['model'])

        # print(self.visual_sentiment)

        # self.dropout = torch.nn.Dropout(0.3)
        self.emotion_layer = torch.nn.Linear(768, 384)

        if opt["label_type"] == "2_way":
            self.category_classifier = torch.nn.Linear(768, 2)  # Bert base 的H是768
            # self.category_classifier = nn.Sequential(
            #     # Layer 1: 768 → 512
            #     nn.Linear(768, 512),
            #     nn.GELU(),  # 比ReLU更平滑的激活函数
            #     nn.LayerNorm(512),  # 稳定训练过程
            #
            #     # Layer 2: 512 → 256
            #     nn.Linear(512, 256),
            #     nn.SiLU(),
            #     nn.Dropout(0.3),  # 防止模态特异性过拟合
            #
            #     # Layer 3: 256 → 2
            #     nn.Linear(256, 2)
            # )
        elif opt["label_type"] == "3_way":
            self.category_classifier = torch.nn.Linear(768, 3)  # Bert base 的H是768
            # self.category_classifier = nn.Sequential(
            #     # Layer 1: 768 → 512
            #     nn.Linear(768, 512),
            #     nn.GELU(),  # 比ReLU更平滑的激活函数
            #     nn.LayerNorm(512),  # 稳定训练过程
            #
            #     # Layer 2: 512 → 256
            #     nn.Linear(512, 256),
            #     nn.SiLU(),
            #     nn.Dropout(0.3),  # 防止模态特异性过拟合
            #
            #     # Layer 3: 256 → 2
            #     nn.Linear(256, 3)
            # )
        else:  # 6_way
            self.category_classifier = torch.nn.Linear(768, 6)  # Bert base 的H是768
            # self.category_classifier = nn.Sequential(
            #     # Layer 1: 768 → 512
            #     nn.Linear(768, 512),
            #     nn.GELU(),  # 比ReLU更平滑的激活函数
            #     nn.LayerNorm(512),  # 稳定训练过程
            #
            #     # Layer 2: 512 → 256
            #     nn.Linear(512, 256),
            #     nn.SiLU(),
            #     nn.Dropout(0.3),  # 防止模态特异性过拟合
            #
            #     # Layer 3: 256 → 2
            #     nn.Linear(256, 6)
            # )

    def forward(self, ids, mask, token_type_ids, pixel_values, pixel_values_emo, emo_ids, emo_mask, intent_ids, intent_mask, labels):
        text_outputs = self.text_encoder(ids, attention_mask=mask, token_type_ids=token_type_ids) # 16*768
        img_outputs = self.img_encoder(pixel_values=pixel_values) # 16*768
        text_lhs = text_outputs.last_hidden_state  # [batch, text_seq_len=100, 768]
        img_lsh = img_outputs.last_hidden_state  # [batch, img_seq_len=197, 768]

        # txt_emo_output = self.text_sentiment(input_ids=emo_ids, attention_mask=emo_mask, output_hidden_states=True) # 13* 16*100*768
        _, txt_emo_embeds = self.text_sentiment(input_ids=emo_ids, attention_mask=emo_mask)
        # vis_emo_output = self.visual_sentiment(pixel_values=pixel_values, output_hidden_states=True) # 之后试一下用pixel_values
        _, vis_emo_embeds = self.visual_sentiment(pixel_values=pixel_values_emo)
        # txt_intent_output = self.intent_detector(input_ids=intent_ids, attention_mask=intent_mask, output_hidden_states=True)
        _, txt_intent_embeds = self.intent_detector(input_ids=intent_ids, attention_mask=intent_mask)

        # 计算 文字图片embedding的cross attention 这里拼接之前乘一个可以学习的weight
        text_embeds, img_embeds = text_lhs[:,0,:], img_lsh[:,0,:]
        # combined_output = torch.cat((text_embeds, img_embeds), dim=1) # 试一下不直接将emotion的embedding拼起来，直接让模型返回text_embed和emo_embed，计算L2loss，让原本的模型得到的text_embed能更好的提取情绪信息 (embedding distillation)
        text_embeds2 = text_lhs
        img_embeds2 = img_lsh[:, 1:, :]
        fused_embed = self.multimodal_block(text_embeds2, img_embeds2)
        output = self.category_classifier(fused_embed)
        return output, text_embeds, img_embeds, txt_emo_embeds, vis_emo_embeds, txt_intent_embeds

