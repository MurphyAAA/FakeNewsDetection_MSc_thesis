# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/24 9:50
@Auth ： Murphy
@File ：load_data.py
@IDE ：PyCharm
"""
import pdb
import PIL
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image, ImageFile
import preprocessing
from torchvision import transforms
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
""" data:
author
clean_title	:移除标点，数字，小写...
created_utc	：创建时间的时间戳
domain	
hasImage	
id	
image_url	
linked_submission_id	
num_comments	
score	
subreddit	
title	
upvote_ratio	
2_way_label	
3_way_label	 0: complete true, 1: sample is fake text is true, 2: sample is fake with false text
6_way_label

对于subreddit为 photoshopbattles 的label全是True
"""


exceptionImages = []


class CustomDataset(Dataset):  # for Bert training
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["2_way_label"]
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):  # 构建dataloader的时候用
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        # 有个问题，text长度不均匀，都padding了浪费资源，怎么能长度不一致也能训练？
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'label': torch.tensor(self.label[index], dtype=torch.long)
        }


class CustomDataset_Clip(Dataset):
    def __init__(self, dataframe, clip_processor, data_path):
        self.clip_processor = clip_processor
        # self.tokenizer = clip.tokenize
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["2_way_label"]
        self.img_id = dataframe["id"]
        self.data_path = data_path

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        # tokenized_text = self.tokenizer(text, truncate=True)
        img_path = f'{self.data_path}/public_image_set/{self.img_id[index]}.jpg'

        # try:
        img = Image.open(img_path)
        # except PIL.UnidentifiedImageError:
        #     exceptionImages.append(self.img_id[index])
        #     return None
        # inputs = self.clip_processor(text=text, images=img, return_tensors="pt", padding="max_length", **{"truncation":True})
        inputs = self.clip_processor(text=text, images=img, return_tensors="pt", padding="max_length", truncation=True)

        ids = torch.squeeze(inputs['input_ids'], dim=0)  # batch_size,77   如果不squeeze去掉最前面的1， 后面拼成batch时会多一个维度
        mask = torch.squeeze(inputs['attention_mask'], dim=0)  # batch_size,77
        pixel_values = torch.squeeze(inputs["pixel_values"], dim=0)  # batch_size,3,224,224

        return {
            'ids': ids.clone().detach(),
            'mask': mask.clone().detach(),
            'pixel_values': pixel_values.clone().detach(),
            # 'label': torch.tensor(self.label[index], dtype=torch.long)
            "label": self.label[index]
        }


def read_file(data_path, filename):
    df = pd.read_csv(f'{data_path}/{filename}.tsv', delimiter='\t')

    # print(df.head())
    return df


def load_dataset(opt):
    df_train = read_file(opt['data_path'], 'multimodal_train')
    df_val = read_file(opt['data_path'], 'multimodal_validate')
    df_test = read_file(opt['data_path'], 'multimodal_test_public')

    df_train = df_train[["clean_title", "id", "2_way_label"]]
    df_val = df_val[["clean_title", "id", "2_way_label"]]
    df_test = df_test[["clean_title", "id", "2_way_label"]]

    df_train_filter, df_val_filter, df_test_filter = preprocessing.filter_image.get_filter_dataset(df_train, df_val, df_test)
    # new_df = df[["subreddit","2_way_label"]]
    # filter_df = new_df[(df["subreddit"]=="photoshopbattles")&(df["2_way_label"]==1)]
    # new_df = filter_df[["subreddit","2_way_label"]]
    # print(filter_df.count())
    # if filter_img_flg == True:
    #     def check_file(filename, dataframe):  # 检查 train,val,test数据集中的图片是否能正常打开，不能打开则将图片名写入文件filename
    #         if not os.path.isfile(f'{opt["data_path"]}/{filename}'):
    #             with open(filename, "w") as file:
    #                 for index in dataframe["id"]:
    #                     img_path = f'{opt["data_path"]}/public_image_set/{index}.jpg'
    #                     try:
    #                         Image.open(img_path)
    #                     except PIL.UnidentifiedImageError:
    #                         file.write(f'{index} ')
    #                         continue
    #         else:
    #             print(f"file {filename} exists")
    #
    #     check_file("filter_image_train.txt", df_train)
    #     check_file("filter_image_val.txt", df_val)
    #     check_file("filter_image_test.txt", df_test)

    return df_train_filter, df_val_filter, df_test_filter


def build_dataloader(opt, clip_processor=None):
    df_train, df_val, df_test = load_dataset(opt)
    print(df_train.head())
    print(f'training set:{df_train.shape}')
    print(f'validation set:{df_val.shape}')
    print(f'testing set:{df_test.shape}')
    # maxlen=0
    # totlen=0
    # for text in df_train["clean_title"]: #看一下text的最大长度
    #     totlen+=len(text.split())
    #     if len(text.split())> maxlen:
    #         maxlen = len(text.split())
    # print(f'max text len:{maxlen}, avg text len:{totlen/df_train.shape[0]}')
    # train max len: 553 words, avg len: 7.5 words # 有个问题，text长度不均匀，都padding了浪费资源，怎么能长度不一致也能训练？ ????????
    if opt["model"] == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        train_set = CustomDataset(df_train, tokenizer, opt['max_len'])
        val_set = CustomDataset(df_val, tokenizer, opt['max_len'])
        test_set = CustomDataset(df_test, tokenizer, opt['max_len'])
    else:  # clip
        train_set = CustomDataset_Clip(df_train, clip_processor, opt['data_path'])
        val_set = CustomDataset_Clip(df_val, clip_processor, opt['data_path'])
        test_set = CustomDataset_Clip(df_test, clip_processor, opt['data_path'])

    train_params = {'batch_size': opt['batch_size'],
                    'num_workers': opt['num_workers'],
                    'shuffle': False}
    val_params = {'batch_size': opt['batch_size'],
                  'num_workers': opt['num_workers'],
                  'shuffle': False}
    test_params = {'batch_size': opt['batch_size'],
                   'num_workers': opt['num_workers'],
                   'shuffle': True}

    train_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    test_loader = DataLoader(test_set, **test_params)
    return train_loader, val_loader, test_loader

# def collate_fn(batch):
#     filtered_batch = [data for data in batch if data is not None]
#     # 如果剩余样本为空，则返回空列表
#     if len(filtered_batch) == 0:
#         return []
#
#     # 否则，构建完整的批次
#     return torch.utils.data._utils.collate.default_collate(filtered_batch)
