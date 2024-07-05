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

exceptionImages = []


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


class CustomDataset_Clip(Dataset):
    def __init__(self, dataframe, clip_processor, data_path):
        self.clip_processor = clip_processor
        # self.tokenizer = clip.tokenize
        self.data = dataframe
        self.text = dataframe["clean_title"]
        self.label = dataframe["label"]
        self.img_id = dataframe["id"]
        self.data_path = data_path

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        img_path = f'{self.data_path}/public_image_set/{self.img_id[index]}.jpg'

        # try:
        img = Image.open(img_path).convert("RGB")
        # print(f"index:{index}, image id:{self.img_id[index]}, image:\n{img}") #看看有问题的image有啥不同！！！！
        # convert_tensor = transforms.ToTensor()
        # tensor = convert_tensor(img)
# 打印张量信息
#         print(f"张量形状: {tensor.shape}")  # 输出： torch.Size([3, 高度, 宽度])
#         print(f"张量数据类型: {tensor.dtype}")  # 输出： torch.float32
        # 尝试一下循环open所有的图片，看看所有的图片是不是都能打开，应该可以。。。vit都行
        # except Image.DecompressionBombWarning:
        #     print(f"图片过大 {self.img_id[index]}")
        # 检查图像的通道数
        # print()
        # if img.mode != "RGB":
        #     raise ValueError(f"图像 {self.img_id[index]} 不是 RGB 模式。实际模式: {img.mode}")
        # print(f"1 {self.img_id[index]}")
        # text only 的解决一下unbalance的问题！！！！
        inputs = self.clip_processor(text=[text], images=img, return_tensors="pt", padding="max_length",
                                     truncation=True)  # (text=text, images=img, return_tensors="pt", padding="max_length", truncation=True)
        # print(f"2 {self.img_id[index]}")
        ids = torch.squeeze(inputs['input_ids'], dim=0)  # batch_size,77   如果不squeeze去掉最前面的1， 后面拼成batch时会多一个维度
        mask = torch.squeeze(inputs['attention_mask'], dim=0)  # batch_size,77
        pixel_values = torch.squeeze(inputs["pixel_values"], dim=0)  # batch_size,3,224,224

        # print(f"{self.img_id[index]} pixel_value:{pixel_values.shape}")
        return {
            'ids': ids.clone().detach(),
            'mask': mask.clone().detach(),
            'pixel_values': pixel_values.clone().detach(),
            # 'label': torch.tensor(self.label[index], dtype=torch.long)
            "label": self.label[index]
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
def read_file(data_path, filename):
    df = pd.read_csv(f'{data_path}/{filename}.tsv', delimiter='\t')

    # print(df.head())
    return df


def load_dataset(opt):
    df_train = read_file(opt['data_path'], 'multimodal_train')[:300]
    df_val = read_file(opt['data_path'], 'multimodal_validate')[:300]
    df_test = read_file(opt['data_path'], 'multimodal_test_public')[:300]
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


def build_dataloader(opt, processor=None):
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
    elif opt["model"] == "clip" or opt["model"] == "clip_large":  # clip/clip_large
        # train_set = CustomDataset_Clip(df_train, processor, opt['data_path'])
        # val_set = CustomDataset_Clip(df_val, processor, opt['data_path'])
        # test_set = CustomDataset_Clip(df_test, processor, opt['data_path'])
        train_set, val_set, test_set = prepare_dataset_clip(opt, processor)
    # elif opt["model"] == "vit" or opt["model"] == "vit_large":
    #     train_set = CustomDataset_Vit(df_train, processor, opt['data_path'])
    #     val_set = CustomDataset_Vit(df_val, processor, opt['data_path'])
    #     test_set = CustomDataset_Vit(df_test, processor, opt['data_path'])
    elif opt["model"] == "bert_vit":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

        train_set, val_set, test_set = prepare_dataset_bert_vit(opt, tokenizer, vit_processor)
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
    texts = [" ".join(str(x.split())) for x in example_batch["clean_title"]]
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
    inputs = vit_processor(images, return_tensors='pt')
    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
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
    texts = [" ".join(str(x.split())) for x in example_batch["clean_title"]]

    images = [Image.open(f'{opt["data_path"]}/public_image_set/{x}.jpg').convert("RGB") for x in
              example_batch['id']]
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