# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import json
import gzip
import os
import random

from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch

PADDING_IDX = 0x100
class AppDataset(Dataset):
    def __init__(self, df, label2idx, train_type):
        assert train_type in ['cls','contr','multi']
        self.train_type = train_type

        self.df = df
        self.label2idx = label2idx
        self.group = {label:df[df['label']==label] for label in label2idx.keys()}

        self.MAXLEN = 28 * 28
        self.padding_idx = PADDING_IDX

    def padding(self,load):
        return torch.tensor(load+[self.padding_idx]*(self.MAXLEN-len(load)))

    def __getitem__(self, item):
        sample = self.df.iloc[item]
        label = sample['label']
        if self.train_type == 'cls':
            return self.padding(sample['upload']), self.padding(sample['download']), torch.tensor(label2idx[label])
        elif self.train_type == 'contr' or self.train_type == 'multi':
            pos_sample = self.group[label].sample().iloc[0]
            while True:
                neg_label = random.choice(list(label2idx.keys()))
                if neg_label != label:
                    break
            neg_sample = self.group[neg_label].sample().iloc[0]
            return self.padding(sample['upload']), self.padding(sample['download']), torch.tensor(label2idx[label]), \
                   self.padding(pos_sample['upload']), self.padding(pos_sample['download']), \
                   self.padding(neg_sample['upload']), self.padding(neg_sample['download'])
        else:
            raise Exception(f'Error train type: {self.train_type}')

    def __len__(self):
        return len(self.df)

class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNN_block, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,padding='same'),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.cnn(x)


class CNN(nn.Module):
    def __init__(self, class_num, out_dim, channels=64,block_num=3):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=256+1,
            embedding_dim=16,
            padding_idx=PADDING_IDX
        )
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=channels, kernel_size=49, padding='same'),
            nn.BatchNorm1d(num_features=channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # 784/4=196
            nn.Dropout(p=0.2)
        )
        blocks, out_len = [], 196
        for i in range(block_num):
            blocks += [
                CNN_block(in_channels=channels, out_channels=channels, kernel_size=3),
                nn.MaxPool1d(kernel_size=2),  # 196/2=98
                nn.Dropout(p=0.2),
            ]
            out_len = int(out_len/2)
        self.blocks = nn.Sequential(*blocks)
        self.dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_len*channels*2, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, out_dim),
        )
        self.cls = nn.Sequential(
            nn.BatchNorm1d(num_features=out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, class_num),
            nn.Softmax(dim=-1)
        )

    def sample_forward(self,x_up,x_down):
        x_up_emb = self.embedding(x_up)
        out_up = self.cnn1(x_up_emb.permute(0, 2, 1))
        out_up = self.blocks(out_up)
        x_down_emb = self.embedding(x_down)
        out_down = self.cnn1(x_down_emb.permute(0, 2, 1))
        out_down = self.blocks(out_down)
        out = torch.concat([out_up, out_down], dim=1)
        return self.dnn(out)

    def sample_prob(self, out):
        p = self.cls(out)
        return p

    def forward(self, x_up, x_down):
        out = self.sample_forward(x_up,x_down)
        p = self.sample_prob(out)
        return p

    def forward_from_embedding(self,x_up_emb,x_dowm_emb):
        out_up = self.cnn(x_up_emb.permute(0, 2, 1))
        out_down = self.cnn(x_dowm_emb.permute(0, 2, 1))
        out = self.dnn(torch.concat([out_up, out_down], dim=1))
        p = self.sample_prob(out)
        return p

    def transfer_features(self, x_up, x_down):
        x_up_emb = self.embedding(x_up)
        out_up = self.cnn1(x_up_emb.permute(0, 2, 1))
        out_up = self.blocks(out_up)
        x_down_emb = self.embedding(x_down)
        out_down = self.cnn1(x_down_emb.permute(0, 2, 1))
        out_down = self.blocks(out_down)
        out = torch.concat([out_up, out_down], dim=1)
        return out


class ResNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResNet_block, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(num_features=out_channels),
        )
    def forward(self,x):
        x_ = self.cnn(x)
        return F.relu(x + x_)

class ResNet(nn.Module):
    def __init__(self, class_num, out_dim, channels=64, block_num=3):
        super(ResNet, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=256+1,
            embedding_dim=16,
            padding_idx=PADDING_IDX
        )
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=channels, kernel_size=49, padding='same'),
            nn.BatchNorm1d(num_features=channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),  # 784/4=196
            nn.Dropout(p=0.2)
        )
        # 196->98->49
        blocks, out_len = [], 196
        for i in range(block_num):
            blocks += [
                ResNet_block(in_channels=channels, out_channels=channels, kernel_size=3),
                nn.MaxPool1d(kernel_size=2),  # 196/2=98
                nn.Dropout(p=0.2),
            ]
            out_len = int(out_len/2)
        self.blocks = nn.Sequential(*blocks)
        self.dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_len*channels*2, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, out_dim),
        )
        self.cls = nn.Sequential(
            nn.BatchNorm1d(num_features=out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, class_num),
            nn.Softmax(dim=-1)
        )

    def sample_forward(self,x_up,x_down):
        x_up_emb = self.embedding(x_up)
        out_up = self.cnn1(x_up_emb.permute(0, 2, 1))
        out_up = self.blocks(out_up)
        x_down_emb = self.embedding(x_down)
        out_down = self.cnn1(x_down_emb.permute(0, 2, 1))
        out_down = self.blocks(out_down)
        out = torch.concat([out_up, out_down], dim=1)
        return self.dnn(out)

    def sample_prob(self, out):
        p = self.cls(out)
        return p

    def forward(self, x_up, x_down):
        out = self.sample_forward(x_up,x_down)
        p = self.sample_prob(out)
        return p

    def forward_from_embedding(self,x_up_emb,x_down_emb):
        out_up = self.cnn1(x_up_emb.permute(0, 2, 1))
        out_up = self.blocks(out_up)
        out_down = self.cnn1(x_down_emb.permute(0, 2, 1))
        out_down = self.blocks(out_down)
        out = self.dnn(torch.concat([out_up, out_down], dim=1))
        p = self.sample_prob(out)
        return p

    def transfer_features(self, x_up, x_down):
        x_up_emb = self.embedding(x_up)
        out_up = self.cnn1(x_up_emb.permute(0, 2, 1))
        out_up = self.blocks(out_up)
        x_down_emb = self.embedding(x_down)
        out_down = self.cnn1(x_down_emb.permute(0, 2, 1))
        out_down = self.blocks(out_down)
        out = torch.concat([out_up, out_down], dim=1)
        return out



from collections import Counter
from tqdm import tqdm
def train(model, iterator, optimizer, train_type, criterion, metrics, clip, device):
    model.train()
    epoch_loss, epoch_acc = Counter(), 0.0
    pbar = tqdm(iterator, desc=f'Epoch: {epoch + 1}', unit='batch', unit_scale=True)
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        if train_type == 'cls':
            upload, download, label = data
            upload, download, label = upload.to(device), download.to(device), label.to(device)
            p = model(upload, download)
            loss = criterion(p, label)
            acc = metrics(p, label)
        elif train_type == 'contr':
            upload, download, label, pos_up, pos_down, neg_up, neg_down = data
            upload, download, label = upload.to(device), download.to(device), label.to(device)
            pos_up, pos_down, neg_up, neg_down = pos_up.to(device), pos_down.to(device), neg_up.to(device), neg_down.to(device)
            out = model.sample_forward(upload,download)
            out_pos = model.sample_forward(pos_up,pos_down)
            out_neg = model.sample_forward(neg_up,neg_down)
            loss = criterion(out, out_pos, out_neg)
        elif train_type == 'multi':
            upload, download, label, pos_up, pos_down, neg_up, neg_down = data
            upload, download, label = upload.to(device), download.to(device), label.to(device)
            pos_up, pos_down, neg_up, neg_down = pos_up.to(device), pos_down.to(device), neg_up.to(device), neg_down.to(device)
            out = model.sample_forward(upload,download)
            p = model.sample_prob(out)
            out_pos = model.sample_forward(pos_up,pos_down)
            out_neg = model.sample_forward(neg_up,neg_down)
            loss = criterion(p, label, out, out_pos, out_neg)
            acc = metrics(p, label)
        else:
            raise Exception(f'Error train type: {train_type}')

        loss['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_item = dict(zip(loss,map(float,loss.values())))
        epoch_loss = epoch_loss + Counter(loss_item)
        if train_type == 'cls' or train_type == 'multi':
            epoch_acc += acc.item()
            pbar.set_postfix(ordered_dict=loss_item, acc=f'{acc.item():.4f}')
        else:
            pbar.set_postfix(ordered_dict=loss_item)

        if i == len(pbar)-1:
            epoch_loss = {key:epoch_loss[key]/len(iterator) for key in epoch_loss.keys()}
            if train_type == 'cls' or train_type == 'multi':
                epoch_acc = epoch_acc/len(iterator)
                pbar.set_postfix(ordered_dict=epoch_loss, acc=f'{epoch_acc:.4f}')
            else:
                pbar.set_postfix(ordered_dict=epoch_loss)
    if train_type == 'contr':
        return epoch_loss
    else:
        return epoch_loss, epoch_acc


def evaluate(model, iterator, train_type, criterion, metrics, device):
    model.eval()
    epoch_loss, epoch_acc = Counter(), 0.0
    with torch.no_grad():
        for i, data in enumerate(iterator):
            if train_type == 'cls':
                upload, download, label = data
                upload, download, label = upload.to(device), download.to(device), label.to(device)
                p = model(upload, download)
                loss = criterion(p, label)
                acc = metrics(p, label)
            elif train_type == 'contr':
                upload, download, label, pos_up, pos_down, neg_up, neg_down = data
                upload, download, label = upload.to(device), download.to(device), label.to(device)
                pos_up, pos_down, neg_up, neg_down = pos_up.to(device), pos_down.to(device), neg_up.to(device), neg_down.to(device)
                out = model.sample_forward(upload, download)
                out_pos = model.sample_forward(pos_up, pos_down)
                out_neg = model.sample_forward(neg_up, neg_down)
                loss = criterion(out, out_pos, out_neg)
            elif train_type == 'multi':
                upload, download, label, pos_up, pos_down, neg_up, neg_down = data
                upload, download, label = upload.to(device), download.to(device), label.to(device)
                pos_up, pos_down, neg_up, neg_down = pos_up.to(device), pos_down.to(device), neg_up.to(device), neg_down.to(device)
                out = model.sample_forward(upload, download)
                p = model.sample_prob(out)
                out_pos = model.sample_forward(pos_up, pos_down)
                out_neg = model.sample_forward(neg_up, neg_down)
                loss = criterion(p, label, out, out_pos, out_neg)
                acc = metrics(p, label)
            else:
                raise Exception(f'Error train type: {train_type}')

            loss_item = dict(zip(loss, map(float, loss.values())))
            epoch_loss = epoch_loss + Counter(loss_item)
            if train_type == 'cls' or train_type == 'multi':
                epoch_acc += acc.item()
    epoch_loss = {key: epoch_loss[key] / len(iterator) for key in epoch_loss.keys()}
    if train_type == 'cls' or train_type == 'multi':
        epoch_acc = epoch_acc / len(iterator)
        return epoch_loss, epoch_acc
    else:
        return epoch_loss



def cls_loss(p, label):
    loss_c = F.cross_entropy(p,label)
    return {'loss':loss_c}
def contr_loss(out, out_pos, out_neg):
    loss_triplet = F.triplet_margin_loss(anchor=out, positive=out_pos, negative=out_neg,)
    return {'loss':loss_triplet}
def multi_loss(p, label, out, out_pos, out_neg):
    loss_c = F.cross_entropy(p,label)
    loss_triplet = F.triplet_margin_loss(anchor=out, positive=out_pos, negative=out_neg,)
    loss = loss_c + loss_triplet
    return {'loss':loss, 'cls':loss_c, 'triplet':loss_triplet}

def accuracy(pred,label):
    return (pred.argmax(1) == label).float().mean()


from sklearn.model_selection import train_test_split
from datetime import datetime
import time

import argparse

# DEBUG = True
DEBUG = False

dataset_name = '2100'
test_size = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--minepoch', type=int, default=20)
    parser.add_argument('--maxwait', type=int, default=5)
    parser.add_argument('--outdim', type=int, default=16)
    parser.add_argument('--blocknum', type=int, default=3)
    parser.add_argument('--channels', type=int, default=64)
    parser.add_argument('--modeltype', type=str, default='CNN')
    parser.add_argument('--traintype', type=str, default='contr')

    args = parser.parse_args()
    BATCH_SIZE = args.batch
    minepoch = args.minepoch
    max_wait = args.maxwait
    out_dim = args.outdim
    block_num = args.blocknum
    channels = args.channels
    model_type = args.modeltype
    train_type = args.traintype

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    with open(f'Dataset/label_{dataset_name}.info', encoding='utf-8') as f:
        label2idx = eval(f.read())
    labels = list(label2idx.keys())
    # print(label2idx)
    with gzip.open(f'Dataset/dataset_{dataset_name}.json.gz') as f:
        DATA = json.loads(f.read())
    df = pd.DataFrame(DATA)
    print(len(df))

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42)
    print(len(df_train),len(df_test))

    trainset = AppDataset(df=df_train, label2idx=label2idx, train_type=train_type)
    testset = AppDataset(df=df_test, label2idx=label2idx, train_type=train_type)
    trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True)
    testloader = DataLoader(testset,batch_size=BATCH_SIZE,shuffle=True)

    if model_type == 'CNN':
        model = CNN(
            class_num=len(labels), out_dim=out_dim,
            channels=channels, block_num=block_num
        ).to(device)
    elif model_type == 'ResNet':
        model = ResNet(
            class_num=len(labels), out_dim=out_dim,
            channels=channels, block_num=block_num
        ).to(device)
    else:
        raise Exception(f'Error model type: {model_type}')

    if train_type == 'cls':
        criterion = cls_loss
    elif train_type == 'contr':
        criterion = contr_loss
    elif train_type == 'multi':
        criterion = multi_loss
    else:
        raise Exception(f'Error train type: {train_type}')

    metrics = accuracy
    optimizer = optim.Adam(model.parameters())

    if DEBUG:
        print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print(f"{total/1024}K")

    N_EPOCHS = 1000
    CLIP = 1
    best_valid_loss = float('inf')

    model_name = f"{train_type}Text-{model_type}"
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    if not DEBUG:
        if os.path.exists(f"model/{model_name}_{timestamp}"):
            print("Sleep 1min")
            time.sleep(60)
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
        os.mkdir(f"model/{model_name}_{timestamp}")
        print(f"{model_name}_{timestamp}")
        with open(f"model/{model_name}_{timestamp}/model.info", 'w') as f:
            f.write(str(model))
        with open(f"model/{model_name}_{timestamp}/label.info", 'w', encoding='utf-8') as f:
            f.write(str(label2idx))

    time.sleep(1)

    history = []
    not_reduce = 0
    for epoch in range(N_EPOCHS):
        if train_type == 'cls' or train_type == 'multi':
            start_time = datetime.now()
            train_loss, train_acc = train(model, trainloader, optimizer, train_type, criterion, metrics, CLIP, device)
            end_time = datetime.now()
            seconds = (end_time - start_time).total_seconds()
            valid_loss, val_acc = evaluate(model, testloader, train_type, criterion, metrics, device)
            print('\tValidation: ', ' '.join([f'valid_{key}={valid_loss[key]:.3f}' for key in valid_loss]), f'val_acc={val_acc:.4f}', end='')
        else:
            start_time = datetime.now()
            train_loss = train(model, trainloader, optimizer, train_type, criterion, metrics, CLIP, device)
            end_time = datetime.now()
            seconds = (end_time - start_time).total_seconds()
            valid_loss = evaluate(model, testloader, train_type, criterion, metrics, device)
            print('\tValidation: ', ' '.join([f'valid_{key}={valid_loss[key]:.3f}' for key in valid_loss]), end='')

        history_item = {'Epoch': epoch + 1, 'Time': seconds}
        history_item.update({f"train_{key}": train_loss[key] for key in train_loss.keys()})
        history_item.update({f"val_{key}": valid_loss[key] for key in valid_loss.keys()})
        if train_type == 'cls' or train_type == 'multi':
            history_item.update({"train_acc": train_acc})
            history_item.update({"val_acc": val_acc})
        history.append(history_item)

        if valid_loss['loss'] < best_valid_loss:
            print(f"\tSave the best model at epoch {epoch + 1}, delta_val_loss={best_valid_loss-valid_loss['loss']:.6f}")
            best_valid_loss = valid_loss['loss']
            not_reduce = 0
            if not DEBUG:
                torch.save(model.state_dict(), f'model/{model_name}_{timestamp}/best.pt')
        else:
            not_reduce += 1
            if not DEBUG:
                torch.save(model.state_dict(), f'model/{model_name}_{timestamp}/final.pt')
            print()

        if not DEBUG:
            pd.DataFrame(history).set_index('Epoch').to_csv(f'model/{model_name}_{timestamp}/history.csv')

        if not_reduce >= max_wait and epoch >= minepoch:
            print(f"Early Stop at Epoch {epoch}")
            break

        time.sleep(1)

        # valid_loss, val_acc = evaluate(model, valid_iterator, criterion, metrics)

        # print(
        #     # f'\tTime: {seconds:.2f}s',
        #     # '\t'+' '.join([f'train_{key}={train_loss[key]:.3f}' for key in train_loss]), f'train_acc={train_acc:.4f}',
        #     '\tValidation: ',
        #     ' '.join([f'valid_{key}={valid_loss[key]:.3f}' for key in valid_loss]), f'val_acc={val_acc:.4f}'
        # )



    pass
