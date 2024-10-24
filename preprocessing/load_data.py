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
from transformers import BertTokenizer, BertForSequenceClassification, ViTImageProcessor
from PIL import Image, ImageFile
import preprocessing
from datasets import Dataset as D
from torchvision import transforms
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 631770000
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
2_way_label	0:fake, 1:true
3_way_label	 0: complete true, 1: sample is fake text is true, 2: sample is fake with false text
6_way_label

对于subreddit为 photoshopbattles 的label全是True
"""



class CustomDataset(Dataset):  # for Bert training
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["label"]
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


class CustomDataset_Vit(Dataset):
    def __init__(self, dataframe, feature_extractor, data_path):
        self.feature_extractor = feature_extractor
        self.data = dataframe
        self.label = dataframe["label"]
        self.img_id = dataframe["id"]
        self.data_path = data_path
        print(type(self.feature_extractor))
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img_path = f'{self.data_path}/public_image_set/{self.img_id[index]}.jpg'

        # try:
        img = Image.open(img_path).convert("RGB")
        # except Image.DecompressionBombWarning:
        #     print(f"图片过大 {self.img_id[index]}")
        inputs = self.feature_extractor(images=img,
                                        return_tensors="pt")  # (text=text, images=img, return_tensors="pt", padding="max_length", truncation=True)
        inputs["labels"] = self.label[index]
        return inputs

class CustomDataset_Bert_Vit(Dataset):
    def __init__(self, dataframe, bert_tokenizer, max_len, vit_processor, data_path):
        self.data = dataframe
        self.bert_tokenizer = bert_tokenizer
        self.max_len = max_len
        self.vit_processor = vit_processor
        self.text = dataframe["clean_title"]
        self.label = dataframe["label"]
        self.img_id = dataframe["id"]
        self.data_path = data_path

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.bert_tokenizer(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        img_path = f'{self.data_path}/public_image_set/{self.img_id[index]}.jpg'
        img = Image.open(img_path).convert("RGB")
        try:
            inputs = self.vit_processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].clone().detach()
            pixel_values = torch.squeeze(pixel_values, dim=0)
        except ValueError as e:
            print(f"Error processing image {img_path}: {e}")
            raise
        return {
            "pixel_values": pixel_values,
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": self.label[index]
        }


class CustomDataset_Albef(Dataset):
    def __init__(self, dataframe,txt_processor, img_processor, data_path):
        self.data = dataframe
        # self.transform = transform

        self.txt_processor = txt_processor
        self.img_processor = img_processor
        self.text = dataframe["clean_title"]
        self.label = dataframe["label"]
        self.img_id = dataframe["id"]
        self.data_path = data_path
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        # pdb.set_trace()
        text = str(self.text[index])
        text = " ".join(text.split())
        text_input = text#self.text_processor["train"](text)
        text_input = self.txt_processor(text_input)
        # pdb.set_trace()
        img_path = f'{self.data_path}/public_image_set/{self.img_id[index]}.jpg'
        img = Image.open(img_path).convert("RGB")
        try:
            # image = self.transform(img)#self.img_processor["train"](img)#.unsqueeze(0)
            image = self.img_processor(img)
        except ValueError as e:
            print(f"Error processing image {img_path}: {e}")
            raise
        # print(image.shape)
        # print(text_input)
        # pdb.set_trace()
        return {
            "image": image,
            "text_input": text_input,
            "labels": torch.tensor(self.label[index], dtype=torch.long)

        }
def read_file(data_path, filename):
    df = pd.read_csv(f'{data_path}/{filename}.tsv', delimiter='\t')

    # print(df.head())
    return df


def load_dataset(opt):
    df_train = read_file(opt['data_path'], 'multimodal_train')#[:800]
    df_val = read_file(opt['data_path'], 'multimodal_validate')#[:300]
    df_test = read_file(opt['data_path'], 'multimodal_test_public')#[:800]
    if opt["label_type"] == "2_way":
        df_train = df_train[["clean_title", "id", "2_way_label"]]
        df_val = df_val[["clean_title", "id", "2_way_label"]]
        df_test = df_test[["clean_title", "id", "2_way_label"]]

        df_train = df_train.rename(columns={"2_way_label": "label"})
        df_val = df_val.rename(columns={"2_way_label": "label"})
        df_test = df_test.rename(columns={"2_way_label": "label"})

    elif opt["label_type"] == "3_way":
        df_train = df_train[["clean_title", "id", "3_way_label"]]
        df_val = df_val[["clean_title", "id", "3_way_label"]]
        df_test = df_test[["clean_title", "id", "3_way_label"]]

        df_train = df_train.rename(columns={"3_way_label": "label"})
        df_val = df_val.rename(columns={"3_way_label": "label"})
        df_test = df_test.rename(columns={"3_way_label": "label"})
    else:  # 6_way
        df_train = df_train[["clean_title", "id", "6_way_label"]]
        df_val = df_val[["clean_title", "id", "6_way_label"]]
        df_test = df_test[["clean_title", "id", "6_way_label"]]

        df_train = df_train.rename(columns={"6_way_label": "label"})
        df_val = df_val.rename(columns={"6_way_label": "label"})
        df_test = df_test.rename(columns={"6_way_label": "label"})

    df_train_filter, df_val_filter, df_test_filter = preprocessing.filter_image.get_filter_dataset(df_train, df_val, df_test)
    return df_train_filter, df_val_filter, df_test_filter


def build_dataloader(opt, processor=None):
    df_train, df_val, df_test = load_dataset(opt)
    print(df_train.head())
    print(f'training set:{df_train.shape}')
    print(f'validation set:{df_val.shape}')
    print(f'testing set:{df_test.shape}')
    train_class_count = torch.bincount(torch.tensor(df_train['label']))
    tot_samples = df_train['label'].shape[0]
    train_class_weights = tot_samples/(int(opt["label_type"][0])*train_class_count)
    print("train", train_class_count)
    print("val  ", torch.bincount(torch.tensor(df_val['label'])))

    # maxlen=0
    # totlen=0
    # for text in df_train["clean_title"]: #看一下text的最大长度
    #     totlen+=len(text.split())
    #     if len(text.split())> maxlen:
    #         maxlen = len(text.split())
    # print(f'max text len:{maxlen}, avg text len:{totlen/df_train.shape[0]}')
    # train max len: 553 words, avg len: 7.5 words # 有个问题，text长度不均匀，都padding了浪费资源，怎么能长度不一致也能训练？ ????????
    if opt["model"] == "bert":
        train_set = CustomDataset(df_train, processor, opt['max_len'])
        val_set = CustomDataset(df_val, processor, opt['max_len'])
        test_set = CustomDataset(df_test, processor, opt['max_len'])
    elif opt["model"] == "clip" or opt["model"] == "clip_large":  # clip/clip_large
        train_set, val_set, test_set = prepare_dataset_clip(opt, processor)
    # elif opt["model"] == "vit" or opt["model"] == "vit_large":
    #     train_set = CustomDataset_Vit(df_train, processor, opt['data_path'])
    #     val_set = CustomDataset_Vit(df_val, processor, opt['data_path'])
    #     test_set = CustomDataset_Vit(df_test, processor, opt['data_path'])
    elif opt["model"] == "bert_vit":
        tokenizer, vit_processor = processor
        train_set, val_set, test_set = prepare_dataset_bert_vit(opt, tokenizer, vit_processor)

    elif opt["model"] == "albef":
        text_processor, img_processor = processor
        # text_processor["eval"] 不能传text_processor，因为是个字典，不能pickle
        train_set = CustomDataset_Albef(df_train, text_processor["eval"], img_processor["eval"], opt['data_path'])
        val_set = CustomDataset_Albef(df_val, text_processor["eval"], img_processor["eval"], opt['data_path'])
        test_set = CustomDataset_Albef(df_test, text_processor["eval"], img_processor["eval"], opt['data_path'])
        # train_set, val_set, test_set = prepare_dataset_albef(opt)
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
    return train_loader, val_loader, test_loader, train_class_weights

def prepare_dataset(opt, processor):
    df_train, df_val, df_test = load_dataset(opt)

    def transform(example_batch):
        # Take a list of PIL images and turn them to pixel values
        images = [Image.open(f'{opt["data_path"]}/public_image_set/{x}.jpg').convert("RGB") for x in
                  example_batch['id']]
        inputs = processor(images, return_tensors='pt')
        # Don't forget to include the labels!
        inputs['labels'] = example_batch['label']
        return inputs

    train_set = D.from_pandas(df_train)
    val_set = D.from_pandas(df_val)
    test_set = D.from_pandas(df_test)

    train_set = train_set.with_transform(transform)
    val_set = val_set.with_transform(transform)
    test_set = test_set.with_transform(transform)

    return train_set, val_set, test_set

def transform_bert_vit(example_batch, bert_processor, vit_processor, opt):
    texts = [" ".join(x.split()) for x in example_batch["clean_title"]]
    inputs=bert_processor(texts, None,
        add_special_tokens=True,
        max_length=opt["max_len"],
        padding="max_length",
        return_token_type_ids=True,
        truncation=True)
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    token_type_ids = inputs["token_type_ids"]

    # Take a list of PIL images and turn them to pixel values
    images = [Image.open(f'{opt["data_path"]}/public_image_set/{x}.jpg').convert("RGB") for x in
              example_batch['id']]
    # convert_tensor = transforms.ToTensor()
    # image_shape = [convert_tensor(x).shape for x in images]
    # print(image_shape)
    image_type = [isinstance(img, PIL.Image.Image) for img in images]
    # print(example_batch['id'], image_type)
    inputs = vit_processor(images, return_tensors='pt')
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    # pdb.set_trace()
    inputs['ids'] = torch.tensor(ids, dtype=torch.long)
    inputs['mask'] = torch.tensor(mask, dtype=torch.long)
    inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
    return inputs
def prepare_dataset_bert_vit(opt, bert_processor, vit_processor):
    df_train, df_val, df_test = load_dataset(opt)
    # 使用 functools.partial 固定住其他参数，只保留 example_batch
    from functools import partial
    transform_fn = partial(transform_bert_vit, bert_processor=bert_processor, vit_processor=vit_processor, opt=opt)

    train_set = D.from_pandas(df_train)
    val_set = D.from_pandas(df_val)
    test_set = D.from_pandas(df_test)

    train_set = train_set.with_transform(transform_fn)
    val_set = val_set.with_transform(transform_fn)
    test_set = test_set.with_transform(transform_fn)

    return train_set, val_set, test_set

def transform_clip(example_batch, processor, opt):
    texts = [" ".join(x.split()) for x in example_batch["clean_title"]]

    images = [Image.open(f'{opt["data_path"]}/public_image_set/{x}.jpg').convert("RGB") for x in
              example_batch['id']]
    # image_type=[isinstance(img, PIL.Image.Image) for img in images]
    # print(example_batch['id'], image_type)
    inputs = processor(text=texts, images=images, return_tensors="pt", padding="max_length",
                                     truncation=True)

    inputs["label"]=example_batch['label']
    return inputs
    # return {
    #     'ids': inputs['input_ids'].clone().detach(),
    #     'mask': inputs['attention_mask'].clone().detach(),
    #     'pixel_values': inputs["pixel_values"].clone().detach(),
    #     "label": example_batch['label']
    # }
def prepare_dataset_clip(opt, processor):
    df_train, df_val, df_test = load_dataset(opt)
    # 使用 functools.partial 固定住其他参数，只保留 example_batch
    from functools import partial
    transform_fn = partial(transform_clip, processor=processor, opt=opt)

    train_set = D.from_pandas(df_train)
    val_set = D.from_pandas(df_val)
    test_set = D.from_pandas(df_test)

    train_set = train_set.with_transform(transform_fn)
    val_set = val_set.with_transform(transform_fn)
    test_set = test_set.with_transform(transform_fn)

    return train_set, val_set, test_set