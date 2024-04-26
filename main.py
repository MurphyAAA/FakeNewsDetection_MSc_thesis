import torch

from parse_args import parse_arguments
from preprocessing.load_data import build_dataloader
from models.base_model import BERTClass
from sklearn import metrics
import pdb

"""
之后用 bert、clip提取特征，fine-tune clip、bert
"""


def main(opt):
    train_loader, val_loader, test_loader = build_dataloader(opt)

    device = torch.device('cpu' if opt["cpu"] else 'cuda:0')

    model = BERTClass()
    # model() 调用__call__()
    model.to(device)

    # define loss function and optimizer
    loss_fun = torch.nn.CrossEntropyLoss()
    loss_fun2 = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=opt['lr'])

    def train(epoch):
        tot_loss=0
        print_loss=0
        model.train()
        for idx, databatch in enumerate(train_loader):# batch_size=32, 共564000个训练样本，17625个batch，循环17625次

            # unpack from dataloader
            ids = databatch["ids"].to(device, dtype=torch.long)
            mask = databatch["mask"].to(device, dtype=torch.long)
            token_type_ids = databatch["token_type_ids"].to(device, dtype=torch.long)
            label = databatch["label"].to(device, dtype=torch.long)
            # predict
            logits = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fun2(logits, label)
            # for visualization
            tot_loss += loss.item()
            print_loss += loss.item()
            if idx % opt["print_every"] == 0:
                print(f"Epoch: {epoch}, batch: {len(train_loader)+1}/{idx+1}, avg_loss: {tot_loss/(idx+1)}, loss_per_{opt['print_every']}: {print_loss/opt['print_every']}") # 打印从训练开始到现在的平均loss，以及最近 "print_every" 次的平均loss
                print_loss = 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ids % opt["val_every"] == 0:
                outputs, labels = validate()
                acc = metrics.accuracy_score(labels, outputs)
                print(f'**acc** : 【{acc*10000//1/100}%】')
                model.train()
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
        train(epoch)

    # outputs, labels = validate()
    # acc = metrics.accuracy_score(labels, outputs)
    # print(f'acc : {acc*10000//1/100}%')

if __name__ == '__main__':
    opt = parse_arguments()
    main(opt)
    print("----finish----")
