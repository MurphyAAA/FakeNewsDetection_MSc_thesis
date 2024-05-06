import torch

from parse_args import parse_arguments
from preprocessing.load_data import build_dataloader
from experiments.bert_exper import BertExperiment
from experiments.clip_exper import ClipExperiment
from sklearn import metrics
import time
import pdb

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

        outputs, labels = experiment.validate()
        # precision=TP/(TP+FP)  recall=TP/(TP+FN)  F1 score  FPR=FP/(FP+TN)
        acc = metrics.accuracy_score(labels, outputs)
        # precision = metrics.precision_score(labels, outputs) # 2-way
        precision = metrics.precision_score(labels, outputs, average=None)  # 3/6-way
        recall = metrics.recall_score(labels, outputs, average=None)  # 3/6-way
        f1 = metrics.f1_score(labels, outputs, average=None)  # 3/6-way
        conf_matrix = metrics.confusion_matrix(labels, outputs)
        # TN = conf_matrix[0,0]
        # FP = conf_matrix[0,1]
        # FPR = FP/(FP+TN)
        FPR = (conf_matrix[1, 0] + conf_matrix[2, 0]) / (conf_matrix.sum() - conf_matrix.diagonal().sum())
        print("—————————— RESULT ——————————")
        print(f'**acc** :       【{acc * 100:.2f}%】')
        print(f'**precision** : 【{precision}】')
        print(f'**recall** :    【{recall}】')
        print(f'**f1** :        【{f1}】')
        print(f'**conf_matrix**:\n【{conf_matrix}】')
        print(f'**FPR** :       【{FPR}】')

    else: # clip
        experiment = ClipExperiment(opt)
        train_loader, val_loader, test_loader = build_dataloader(opt, experiment.preprocess)
        experiment.set_dataloader(train_loader, val_loader, test_loader)

        for epoch in range(opt['num_epochs']):
            epoch_time = experiment.train(epoch)
            print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")

if __name__ == '__main__':
    opt = parse_arguments()
    main(opt)
    print("----finish----")
