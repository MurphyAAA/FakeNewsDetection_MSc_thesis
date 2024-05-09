# -*- coding: utf-8 -*-
"""
@Time ： 2024/4/24 15:54
@Auth ： Murphy
@File ：check_image.py
@IDE ：PyCharm
"""
import os
import PIL
from PIL import Image
import pandas as pd


def read_file(data_path, filename):
    df = pd.read_csv(f'{data_path}/{filename}.tsv', delimiter='\t')

    # print(df.head())
    return df


def check_file(filename, dataframe):  # 检查 train,val,test数据集中的图片是否能正常打开，不能打开则将图片名写入文件filename
    if not os.path.isfile(f'data/Fakeddit/{filename}'):
        with open(filename, "w") as file:
            print(f"starting check 【{filename}】")
            print(os.getcwd())
            for index in dataframe["id"]:
                img_path = f'../data/Fakeddit/public_image_set/{index}.jpg'
                try:
                    Image.open(img_path)
                except PIL.UnidentifiedImageError:
                    file.write(f'{index} ')
                    continue
    else:
        print(f"file {filename} exists")


if __name__ == '__main__':
    df_train = read_file("../data/Fakeddit", 'multimodal_train')
    df_val = read_file("../data/Fakeddit", 'multimodal_validate')
    df_test = read_file("../data/Fakeddit", 'multimodal_test_public')

    df_train = df_train[["clean_title", "id", "2_way_label"]]
    df_val = df_val[["clean_title", "id", "2_way_label"]]
    df_test = df_test[["clean_title", "id", "2_way_label"]]

    check_file("filter_image_train.txt", df_train)
    check_file("filter_image_val.txt", df_val)
    check_file("filter_image_test.txt", df_test)
    print("---end---")
