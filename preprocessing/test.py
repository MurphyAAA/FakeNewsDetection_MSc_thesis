# -*- coding: utf-8 -*-
"""
@Time ： 2024/5/11 11:15
@Auth ： Murphy
@File ：test.py
@IDE ：PyCharm
"""

from load_data import read_file

def read_image_id(filename):
    with open(f"./data/Fakeddit/failed_images/{filename}", "r") as file:
        content = file.read()
    ids = content.split("\'")[1::2]
    id_list = [id.strip("'")for id in ids]
    return id_list

def get_filter_dataset(df_train, df_val, df_test):
    id_list_train = read_image_id("failed_images_train.txt")
    id_list_val = read_image_id("failed_images_val.txt")
    id_list_test = read_image_id("failed_images_test.txt")

    filter_df_train = df_train[~df_train["id"].isin(id_list_train)]
    filter_df_val = df_val[~df_val["id"].isin(id_list_val)]
    filter_df_test = df_test[~df_test["id"].isin(id_list_test)]

    return filter_df_train, filter_df_val, filter_df_test

# if __name__ == "__main__":
#
#     id_list_train = read_image_id("failed_images_train.txt")
#     id_list_val = read_image_id("failed_images_val.txt")
#     id_list_test = read_image_id("failed_images_test.txt")
#
#     df_train = read_file("./data/Fakeddit/", 'multimodal_train')
#     df_val = read_file("./data/Fakeddit/", 'multimodal_validate')
#     df_test = read_file("./data/Fakeddit/", 'multimodal_test_public')
#
#     filter_df_train = df_train[~df_train["id"].isin(id_list_train)]
#     filter_df_val = df_val[~df_val["id"].isin(id_list_val)]
#     filter_df_test = df_test[~df_test["id"].isin(id_list_test)]