# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/24 9:50
@Auth ： Murphy
@File ：load_data.py
@IDE ：PyCharm
"""

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
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from PIL import Image


class CustomDataset(Dataset):  # for Bert training
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["6_way_label"]
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
    def __init__(self, dataframe, clip_preprocess, data_path):
        self.clip_preprocess = clip_preprocess
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["6_way_label"]
        self.img_id = dataframe["id"]
        self.data_path = data_path

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        img_path = f'{self.data_path}/public_image_set/{self.img_id[index]}.jpg'
        # img_path, descriptions = self.examples[index]

        img = self.clip_preprocess(Image.open(img_path))
        # x, descriptions  # tuple(image, descriptions)
        return {
            "img": img,
            "description": text,
            "label": self.label[index]
        }


def read_file(data_path, filename):
    df = pd.read_csv(f'{data_path}/{filename}.tsv', delimiter='\t')

    # print(df.head())
    return df


def load_datset(opt):
    df_train = read_file(opt['data_path'], 'multimodal_train')
    df_val = read_file(opt['data_path'], 'multimodal_validate')
    df_test = read_file(opt['data_path'], 'multimodal_test_public')

    df_train = df_train[["clean_title", "id", "6_way_label"]]
    df_val = df_val[["clean_title", "id", "6_way_label"]]
    df_test = df_test[["clean_title", "id", "6_way_label"]]
    # new_df = df[["subreddit","2_way_label"]]
    # filter_df = new_df[(df["subreddit"]=="photoshopbattles")&(df["2_way_label"]==1)]
    # new_df = filter_df[["subreddit","2_way_label"]]
    # print(filter_df.count())

    return df_train, df_val, df_test


def build_dataloader(opt):
    df_train, df_val, df_test = load_datset(opt)
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
    else: # clip
        train_set = CustomDataset_Clip(df_train,d)

    train_params = {'batch_size': opt['batch_size'],
                    'num_workers': opt['num_workers'],
                    'shuffle': True}
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
