import os.path

import torch

from parse_args import parse_arguments
from preprocessing.load_data import build_dataloader
from experiments.bert_exper import BertExperiment
from experiments.clip_exper import ClipExperiment
from sklearn import metrics
import time
import pdb
import sys

"""
之后用 bert、clip提取特征，fine-tune clip、bert
要求的意思是只用bert或clip提取特征么？而不是直接用那个模型？？
输出一下每50个batch的时间，以及训练，验证总时间！
"""


def main(opt):
    if opt["model"] == "bert":
        experiment = BertExperiment(opt)
        train_loader, val_loader, test_loader = build_dataloader(opt)
        experiment.set_dataloader(train_loader, val_loader, test_loader)
        for epoch in range(opt["num_epochs"]):
            epoch_time = experiment.train(epoch)
            print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")

        predicts, labels = experiment.validate()
        # precision=TP/(TP+FP)  recall=TP/(TP+FN)  F1 score  FPR=FP/(FP+TN)
        evaluation(labels,predicts,False)

    else:  # clip
        experiment = ClipExperiment(opt)
        train_loader, val_loader, test_loader = build_dataloader(opt, clip_processor=experiment.processor)
        experiment.set_dataloader(train_loader, val_loader, test_loader)
        if os.path.exists(f'{opt["output_path"]}/checkpoint_epoch_0.pth'):
            print("loading model")
            start_epoch, tot_loss = experiment.load_clip_checkpoint(f'{opt["output_path"]}/checkpoint_epoch_0.pth')
        else:
            start_epoch = 0
            tot_loss = 0
            # train
        # for epoch in range(start_epoch, opt['num_epochs']):
        #     epoch_time, loss = experiment.train(epoch, tot_loss)
        #     experiment.save_clip_checkpoint(f'{opt["output_path"]}/checkpoint_epoch_{epoch}.pth', epoch, loss)
        #     print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")

        predicts, labels = experiment.validation()
        evaluation(labels, predicts, True)

def evaluation(labels, predicts, two_way):
    if two_way: # 2-way
        acc = metrics.accuracy_score(labels, predicts)
        precision = metrics.precision_score(labels, predicts) # 2-way
        precision_macro = metrics.precision_score(labels, predicts, average="macro")
        recall = metrics.recall_score(labels, predicts)  # 3/6-way
        recall_macro = metrics.recall_score(labels, predicts, average="macro")
        f1 = metrics.f1_score(labels, predicts)  # 3/6-way
        f1_macro = metrics.f1_score(labels, predicts, average="macro")
        conf_matrix = metrics.confusion_matrix(labels, predicts)
        TN = conf_matrix[0,0]
        FP = conf_matrix[0,1]
        FPR = FP/(FP+TN)
        print("—————————— RESULT ——————————")
        print(f'**acc** :       【{acc * 100:.2f}%】')
        print(f'**precision** : 【{precision}】------- **precision-Macro** : 【{precision_macro}】')
        print(f'**recall** :    【{recall}】------- **precision-Macro** : 【{recall_macro}】')
        print(f'**f1** :        【{f1}】------- **precision-Macro** : 【{f1_macro}】')
        print(f'**conf_matrix**:\n【{conf_matrix}】')
        print(f'**FPR** :       【{FPR}】')
    else: # 3/6-way
        acc = metrics.accuracy_score(labels, predicts)
        precision = metrics.precision_score(labels, predicts, average=None)  # 3/6-way
        precision_macro = metrics.precision_score(labels, predicts, average="macro")
        recall = metrics.recall_score(labels, predicts, average=None)  # 3/6-way
        recall_macro = metrics.recall_score(labels, predicts, average="macro")
        f1 = metrics.f1_score(labels, predicts, average=None)  # 3/6-way
        f1_macro = metrics.f1_score(labels, predicts, average="macro")
        conf_matrix = metrics.confusion_matrix(labels, predicts)
        # TN = conf_matrix[0,0]
        # FP = conf_matrix[0,1]
        # FPR = FP/(FP+TN)
        FPR = (conf_matrix[1, 0] + conf_matrix[2, 0]) / (
                conf_matrix.sum() - conf_matrix.diagonal().sum())  # how many fake news be trated as true in the false classified cases
        print("—————————— RESULT ——————————")
        print(f'**acc** :       【{acc * 100:.2f}%】')
        print(f'**precision** : 【{precision}】------- **precision-Macro** : 【{precision_macro}】')
        print(f'**recall** :    【{recall}】------- **precision-Macro** : 【{recall_macro}】')
        print(f'**f1** :        【{f1}】------- **precision-Macro** : 【{f1_macro}】')
        print(f'**conf_matrix**:\n【{conf_matrix}】')
        print(f'**FPR** :       【{FPR}】')


if __name__ == '__main__':
    print(sys.path)
    opt = parse_arguments()
    main(opt)
    print("----finish----")
