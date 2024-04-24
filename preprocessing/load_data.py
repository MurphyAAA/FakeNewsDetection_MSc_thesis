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
3_way_label	
6_way_label

对于subreddit为 photoshopbattles 的label全是True
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["2_way_label"]
        self.max_len = max_len
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index): # 构建dataloader的时候用

        inputs = self.tokenizer(
            self.text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
def read_file(data_path,filename):
    df = pd.read_csv(f'{data_path}/{filename}.tsv', delimiter='\t')

    # print(df.head())
    return df

def load_datset(opt):
    df_train = read_file(opt['data_path'], 'multimodal_train')
    df_val = read_file(opt['data_path'], 'multimodal_validate')
    df_test = read_file(opt['data_path'], 'multimodal_test_public')

    df_train = df_train[["clean_title","2_way_label"]]
    df_val = df_val[["clean_title","2_way_label"]]
    df_test = df_test[["clean_title","2_way_label"]]
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
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_set = CustomDataset(df_train, tokenizer, opt['max_len'])
    val_set = CustomDataset(df_val, tokenizer, opt['max_len'])
    test_set = CustomDataset(df_test, tokenizer, opt['max_len'])

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