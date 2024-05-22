# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/15 9:18
@Auth ： Murphy
@File ：clip_model.py
@IDE ：PyCharm
"""
import torch
import transformers
from transformers import CLIPModel, CLIPProcessor


class ClipClass(torch.nn.Module):
    def __init__(self, opt):
        super(ClipClass, self).__init__()
        # self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        if opt["model"] == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:  # clip_large
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # self.l2 = torch.nn.Linear(512,64) # ***********这一层加不加后面再调整
        self.l3 = torch.nn.Dropout(0.2)
        # self.l4 = torch.nn.Linear(128, 2)
        if opt["label_type"] == "2_way":
            self.l4 = torch.nn.Linear(1024, 2)
        elif opt["label_type"] == "3_way":
            self.l4 = torch.nn.Linear(1024, 3)
        else:  # 6_way
            self.l4 = torch.nn.Linear(1024, 6)

    # def __call__(self, *args, **kwargs):
    #     print("call Bert Class")
    def forward(self, ids, mask, pixel_values):
        output_1 = self.model(input_ids=ids, attention_mask=mask,
                              pixel_values=pixel_values)  # 本任务更关注text和img的关系，而不是根据一个分类另一个
        img_embeds, text_embeds = output_1.image_embeds, output_1.text_embeds  # 用这个去训练，不要用logit ！！！！！！！！！！  要用embedding和ground truth直接计算loss 还是要加一个FC层？

        # output_2_img = self.l2(img_embeds)
        # output_2_text = self.l2(text_embeds)
        # combined_output = torch.cat((output_2_text, output_2_img), dim=1)
        combined_output = torch.cat((text_embeds, img_embeds), dim=1)  # ********** 组合方式也可以调整
        output_3 = self.l3(combined_output)
        output = self.l4(output_3)
        return output
