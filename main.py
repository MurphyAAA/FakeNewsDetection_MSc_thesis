import torch

from parse_args import parse_arguments
from preprocessing.load_data import build_dataloader
from models.base_model import BERTClass
from sklearn import metrics
import time
import pdb

"""
之后用 bert、clip提取特征，fine-tune clip、bert
要求的意思是只用bert或clip提取特征么？而不是直接用那个模型？？
输出一下每50个batch的时间，以及训练，验证总时间！
"""


def main(opt):
    train_loader, val_loader, test_loader = build_dataloader(opt)
    device = torch.device('cpu' if opt["cpu"] else 'cuda:0')
    # print(device)

    model = BERTClass()
    # model() 调用__call__()
    model.to(device)

    # define loss function and optimizer
    loss_fun = torch.nn.CrossEntropyLoss()
    loss_fun2 = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt['lr'])

    # return training time
    def train(epoch):
        tot_loss=0
        print_loss=0
        tot_time=0
        model.train()
        start_time = time.time()
        for idx, databatch in enumerate(train_loader):# batch_size=32, 共564000个训练样本，17625个batch，循环17625次
            # unpack from dataloader
            ids = databatch["ids"].to(device, dtype=torch.long)
            mask = databatch["mask"].to(device, dtype=torch.long)
            token_type_ids = databatch["token_type_ids"].to(device, dtype=torch.long)
            label = databatch["label"].to(device, dtype=torch.long)
            # predict
            logits = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            # pdb.set_trace()
            loss = loss_fun(logits, label)
            # for visualization
            tot_loss += loss.item()
            print_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % opt["print_every"] == 0:
                end_time = time.time()
                exe_time =(end_time - start_time)
                tot_time += exe_time
                start_time = time.time()
                print(f"Epoch: {epoch}, batch: {len(train_loader)+1}/{idx+1}, avg_loss: {tot_loss/(idx+1)}, loss_per_{opt['print_every']}: {print_loss/opt['print_every']}, time:{exe_time:.2f}s") # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                print_loss = 0

            # if idx % opt["val_every"] == 0:
            #     outputs, labels = validate()
            #     acc = metrics.accuracy_score(labels, outputs)
            #     print(f'**acc** : 【{acc*10000//1/100}%】')
            #     model.train()
        return tot_time
    # return predict result and real label
    def validate():
        model.eval()
        fin_label=[]
        fin_output=[]
        with torch.no_grad():
            for _, databatch in enumerate(val_loader):
                ids = databatch["ids"].to(device, dtype=torch.long)
                mask = databatch["mask"].to(device, dtype=torch.long)
                token_type_ids = databatch["token_type_ids"].to(device, dtype=torch.long)
                label = databatch["label"]

                logits = model(ids, mask, token_type_ids)
                pred = torch.argmax(logits, dim=-1)
                fin_label.extend(label.cpu().detach().tolist())
                fin_output.extend(pred.cpu().detach().tolist())
                # pdb.set_trace()
        return fin_output, fin_label

    for epoch in range(opt["num_epochs"]):
        tot_time = train(epoch)
        print(f"EPOCH:[{epoch}]  EXECUTION TIME: {tot_time:.2f}s")

    outputs, labels = validate()
    # precision=TP/(TP+FP)  recall=TP/(TP+FN)  F1 score  FPR=FP/(FP+TN)
    acc = metrics.accuracy_score(labels, outputs)
    precision = metrics.precision_score(labels, outputs)
    recall = metrics.recall_score(labels, outputs)
    f1 = metrics.f1_score(labels, outputs)
    conf_matrix = metrics.confusion_matrix(labels, outputs)
    TN = conf_matrix[0,0]
    FP = conf_matrix[0,1]
    FPR = FP/(FP+TN)
    print("—————————— RESULT ——————————")
    print(f'**acc** :       【{acc*100:.2f}%】')
    print(f'**precision** : 【{precision*100:.2f}%】')
    print(f'**recall** :    【{recall*100:.2f}%】')
    print(f'**f1** :        【{f1*100:.2f}%】')
    print(f'**FPR** :       【{FPR*100:.2f}%】')

if __name__ == '__main__':
    opt = parse_arguments()
    main(opt)
    print("----finish----")
