import os.path
import pdb

from parse_args import parse_arguments
from preprocessing.load_data import build_dataloader, prepare_dataset
from experiments.text.bert_exper import BertExperiment
from experiments.clip_exper import ClipExperiment
from experiments.visual.vit_exper import VitExperiment
from experiments.multimodal.bert_vit_exper import Bert_VitExperiment
from experiments.multimodal.albef_exper import AlbefExperiment
from sklearn import metrics
import sys
from transformers import Trainer, TrainingArguments
import numpy as np

"""
之后用 bert、clip提取特征，fine-tune clip、bert
要求的意思是只用bert或clip提取特征么？而不是直接用那个模型？？
输出一下每50个batch的时间，以及训练，验证总时间！
"""
def load_model(experiment):
    start_epoch = 0
    for i in range(opt["num_epochs"], -1, -1):
        fileName = f'{opt["output_path"]}/checkpoint_{opt["model"]}_epoch_{i}_{opt["label_type"]}.pth'
        if os.path.exists(fileName):
            print("loading model")
            start_epoch = experiment.load_checkpoint(fileName)
            break
    return start_epoch

def main(opt):
    if opt["model"] == "bert":
        experiment = BertExperiment(opt)
        train_loader, val_loader, test_loader, train_class_weights = build_dataloader(opt, processor=experiment.tokenizer)
        # experiment.set_weighted_loss(class_weight=train_class_weights)
        experiment.set_dataloader(train_loader, val_loader, test_loader)
        start_epoch = load_model(experiment)
        for epoch in range(start_epoch, opt['num_epochs']):
            epoch_time = experiment.train(epoch)
            experiment.save_checkpoint(
                f'{opt["output_path"]}/checkpoint_{opt["model"]}_epoch_{epoch}_{opt["label_type"]}.pth', epoch)
            print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")
        experiment.writer.close()
        # validation
        predicts, labels = experiment.validate()
        if opt["label_type"] == "2_way":
            result = evaluation(labels, predicts, True)
        else:  # 3/6_way
            result = evaluation(labels, predicts, False)
        print_results(result)
    elif opt["model"] == "clip" or opt["model"] == "clip_large":  # clip/ clip_large
        experiment = ClipExperiment(opt)
        train_loader, val_loader, test_loader, train_class_weights = build_dataloader(opt, processor=experiment.processor)
        experiment.set_weighted_loss(class_weight=train_class_weights)
        experiment.set_dataloader(train_loader, val_loader, test_loader)
        start_epoch = load_model(experiment)
        # train
        for epoch in range(start_epoch, opt['num_epochs']):
            epoch_time = experiment.train(epoch)
            # experiment.save_checkpoint(
            #     f'{opt["output_path"]}/checkpoint_{opt["model"]}_epoch_{epoch}_{opt["label_type"]}.pth', epoch)
            # print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")
        experiment.writer.close()
        # validation
        predicts, labels = experiment.validation()
        if opt["label_type"] == "2_way":
            result = evaluation(labels, predicts, True)
        else:  # 3/6_way
            result = evaluation(labels, predicts, False)
        print_results(result)
    elif opt["model"] == "vit" or opt["model"] == "vit_large":
        experiment = VitExperiment(opt)
        print("training")

        training_args = TrainingArguments(
            output_dir=f'{opt["output_path"]}/vit1_{opt["label_type"]}_new_dataset',
            evaluation_strategy='epoch',
            per_device_train_batch_size=opt["batch_size"],
            per_device_eval_batch_size=opt["batch_size"],
            num_train_epochs=opt["num_epochs"],
            fp16=True,
            save_strategy='epoch',
            logging_dir=f'{opt["output_path"]}/vit1_{opt["label_type"]}_new_dataset',
            logging_steps=opt["print_every"],
            remove_unused_columns=False,
            load_best_model_at_end=True,
        )
        train_set, val_set, test_set = prepare_dataset(opt, experiment.processor)
        trainer = Trainer(
            model=experiment.model,
            args=training_args,
            data_collator=experiment.collate_fn,
            train_dataset=train_set,
            eval_dataset=val_set,
            tokenizer=experiment.processor,
            compute_metrics=compute_metrics,
        )
        train_results = trainer.train()
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()
        metrics = trainer.evaluate(val_set)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    elif opt["model"] == "bert_vit":
        experiment = Bert_VitExperiment(opt)
        processors = (experiment.processors['bert'], experiment.processors['vit'], experiment.processors['text_sentiment'], experiment.processors['visual_sentiment'], experiment.processors['intent_detector'])
        train_loader, val_loader, test_loader, train_class_weights = build_dataloader(opt, processor=processors)
        experiment.set_weighted_loss(class_weight=train_class_weights)
        experiment.set_dataloader(train_loader, val_loader, test_loader)

        start_epoch = load_model(experiment)
        # train
        tot_loss = 0
        for epoch in range(start_epoch, opt['num_epochs']):
            epoch_time, loss = experiment.train(epoch, tot_loss)
            tot_loss = loss
            print(tot_loss)
            # experiment.save_checkpoint(
            #     f'{opt["output_path"]}/checkpoint_{opt["model"]}_epoch_{epoch}_{opt["label_type"]}.pth', epoch)
            # print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")
        # validation
            predicts, labels = experiment.validation()
            if opt["label_type"] == "2_way":
                result = evaluation(labels, predicts, True)
            else:  # 3/6_way
                result = evaluation(labels, predicts, False)
            # log
            # experiment.writer.add_scalar("result_acc", result['acc'], epoch)
            # experiment.writer.add_scalar("result_precision", result['precision_macro'], epoch)
            # experiment.writer.add_scalar("result_recall", result['recall_macro'], epoch)
            # experiment.writer.add_scalar("result_f1", result['f1_macro'], epoch)
        print_results(result)
        # experiment.writer.close()
    elif opt["model"] == "albef":
        experiment = AlbefExperiment(opt)
        train_loader, val_loader, test_loader, train_class_weights = build_dataloader(opt, (experiment.text_processor, experiment.img_processor))
        experiment.set_weighted_loss(class_weight=train_class_weights)
        experiment.set_dataloader(train_loader, val_loader, test_loader)

        start_epoch = load_model(experiment)
        # train
        for epoch in range(start_epoch, opt['num_epochs']):
            epoch_time = experiment.train(epoch)
            experiment.save_checkpoint(
                f'{opt["output_path"]}/checkpoint_{opt["model"]}_epoch_{epoch}_{opt["label_type"]}.pth', epoch)
            print(f"EPOCH:[{epoch}]  EXECUTION TIME: {epoch_time:.2f}s")
        experiment.writer.close()
        # validation
        predicts, labels = experiment.validation()
        if opt["label_type"] == "2_way":
            result = evaluation(labels, predicts, True)
        else:  # 3/6_way
            result = evaluation(labels, predicts, False)
        print_results(result)

def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    accuracy = metrics.accuracy_score(labels, preds)
    # f1 = metrics.f1_score(labels, preds, average=None)
    f1_macro = metrics.f1_score(labels, preds, average='macro')

    # recall = metrics.recall_score(labels, preds, average=None)
    recall_macro = metrics.recall_score(labels, preds, average="macro")

    # precision = metrics.precision_score(labels, preds, average=None)  # 2_way
    precision_macro = metrics.precision_score(labels, preds, average="macro")

    conf_matrix = metrics.confusion_matrix(labels, preds)
    # 2-way
    if opt["label_type"] == '2_way':
        FPR = conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[0,0])# False Positive Rate:  That is, how much false news is mistakenly believed to be true news =错误当成真消息数/假消息总数
    else:
    # 3-way/6-way
        FPR = np.sum(conf_matrix[1:, 0]) / np.sum(conf_matrix[1:, :])
    # FRR = (conf_matrix[1,0]+conf_matrix[2,0]) / (conf_matrix[1,0]+conf_matrix[1,1]+conf_matrix[1,2]+
    #                                              conf_matrix[2,0]+conf_matrix[2,1]+conf_matrix[2,2])
    return {"accuracy": accuracy,
            "f1_marco": f1_macro,
            "recall_macro": recall_macro,
            "precision_macro": precision_macro,
            "False Positive Rate:": FPR
            }


def evaluation(labels, predicts, two_way):
    # precision=TP/(TP+FP)  recall=TP/(TP+FN)  F1 score  FPR=FP/(FP+TN)
    if two_way:  # 2_way
        acc = metrics.accuracy_score(labels, predicts)
        precision = metrics.precision_score(labels, predicts)  # 2_way
        precision_macro = metrics.precision_score(labels, predicts, average="macro")
        recall = metrics.recall_score(labels, predicts)  # 3/6_way
        recall_macro = metrics.recall_score(labels, predicts, average="macro")
        f1 = metrics.f1_score(labels, predicts)  # 3/6_way
        f1_macro = metrics.f1_score(labels, predicts, average="macro")
        conf_matrix = metrics.confusion_matrix(labels, predicts)
        # TN = conf_matrix[0, 0]
        # FP = conf_matrix[0, 1]
        # FN = conf_matrix[1, 0]
        # TP = conf_matrix[1, 1]
        # FPR = FP / (FP + TN)
        # TPR = TP/TP+FN
        #     预 测
        # 真
        # 实
        # FPR = conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0])
        # TPR = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
        FPR, TPR, thresholds = metrics.roc_curve(labels, predicts)
        roc_auc = metrics.auc(FPR, TPR)
    else:  # 3/6_way
        acc = metrics.accuracy_score(labels, predicts)
        precision = metrics.precision_score(labels, predicts, average=None, labels=np.unique(predicts))  # 3/6_way
        precision_macro = metrics.precision_score(labels, predicts, average="macro", labels=np.unique(predicts)) # , labels=np.unique(predicts) 只统计最少被预测过1次的
        recall = metrics.recall_score(labels, predicts, average=None, labels=np.unique(predicts))  # 3/6_way
        recall_macro = metrics.recall_score(labels, predicts, average="macro", labels=np.unique(predicts))
        f1 = metrics.f1_score(labels, predicts, average=None, labels=np.unique(predicts))  # 3/6_way
        f1_macro = metrics.f1_score(labels, predicts, average="macro", labels=np.unique(predicts))
        conf_matrix = metrics.confusion_matrix(labels, predicts)
        FPR = np.sum(conf_matrix[1:, 0]) / np.sum(conf_matrix[1:, :])
    return {
        'acc':acc,
        'precision':precision, 'precision_macro':precision_macro,
        'recall':recall, 'recall_macro':recall_macro,
        'f1':f1, 'f1_macro':f1_macro,
        'conf_matrix':conf_matrix,
        'FPR':FPR,
    }

def print_results(result):
    acc = result['acc']
    precision = result['precision']
    precision_macro = result['precision_macro']
    recall = result['recall']
    recall_macro = result['recall_macro']
    f1 = result['f1']
    f1_macro = result['f1_macro']
    conf_matrix = result['conf_matrix']
    FPR = result['FPR']
    print("—————————— RESULT ——————————")
    print(f'**acc** :       【{acc * 100:.2f}%】')
    print(f'**precision** : 【{precision}】------- **precision-Macro** : 【{precision_macro}】')
    print(f'**recall** :    【{recall}】------- **recall-Macro** : 【{recall_macro}】')
    print(f'**f1** :        【{f1}】------- **f1-Macro** : 【{f1_macro}】')
    print(f'**conf_matrix**:\n【{conf_matrix}】')
    print(f'**FPR** :       【{FPR}】')
    # if two_way:
    #     print(f'**TPR** :       【{TPR}】')
    #     print(f'**ROC_auc** :   【{roc_auc}】')



# def print_to_file(*args, **kwargs):
#     tmp_file = os.path.join(opt["tmp_dir"], 'temp_output.txt')
#     with open(tmp_file, 'w') as temp_file:
#         print(*args, **kwargs, file=temp_file)


if __name__ == '__main__':
    opt = parse_arguments()

    print(sys.path)
    # print(sys.path)
    main(opt)
    # print("----finish----")
    print("----finish----")
