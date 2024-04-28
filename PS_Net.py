# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import random
import json
import gzip
import os

from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
import torch

MAXLEN = 256
class AppDataset(Dataset):
    def __init__(self, df, label2idx, train_type):
        assert train_type in ['cls', 'contr', 'multi']
        self.train_type = train_type

        self.df = df
        self.label2idx = label2idx
        self.group = {label: df[df['label'] == label] for label in label2idx.keys()}

    def __getitem__(self, item):
        sample = self.df.iloc[item]
        label = sample['label']
        if self.train_type == 'cls':
            path = torch.tensor([[p['iat'], p['l'], p['b'], 1 if p['d'] == '>' else -1] for i,p in enumerate(sample['packets'][0:MAXLEN])])
            return (path, len(path)), label2idx[label]
        elif self.train_type == 'contr' or self.train_type == 'multi':
            pos_sample = self.group[label].sample().iloc[0]
            while True:
                neg_label = random.choice(list(label2idx.keys()))
                if neg_label != label:
                    break
            neg_sample = self.group[neg_label].sample().iloc[0]
            path = torch.tensor([[p['iat'], p['l'], p['b'], 1 if p['d'] == '>' else -1] for i,p in enumerate(sample['packets'][0:MAXLEN])])
            pos_path = torch.tensor([[p['iat'], p['l'], p['b'], 1 if p['d'] == '>' else -1] for i,p in enumerate(pos_sample['packets'][0:MAXLEN])])
            neg_path = torch.tensor([[p['iat'], p['l'], p['b'], 1 if p['d'] == '>' else -1] for i,p in enumerate(neg_sample['packets'][0:MAXLEN])])
            return (path,len(path)), label2idx[label], (pos_path,len(pos_path)), (neg_path,len(neg_path))
        else:
            raise Exception(f'Error train type: {self.train_type}')

    def __len__(self):
        return len(self.df)


from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

def collate_fn_cls(data):
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    path, label = zip(*data)
    path, path_len = zip(*path)
    path, label = list(path), list(label)
    path = pad_sequence(path,batch_first=True).float()
    # path = pack_padded_sequence(pad_sequence(path).float(), path_len, enforce_sorted=False)
    return path, torch.tensor(path_len), torch.tensor(label)

def collate_fn_contr(data):
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    path, label, pos_path, neg_path = zip(*data)
    path, path_len = zip(*path)
    pos_path, pos_path_len = zip(*pos_path)
    neg_path, neg_path_len = zip(*neg_path)

    path, label, pos_path, neg_path = list(path), list(label), list(pos_path), list(neg_path)

    path = pad_sequence(path,batch_first=True).float()
    pos_path = pad_sequence(pos_path,batch_first=True).float()
    neg_path = pad_sequence(neg_path,batch_first=True).float()
    # path = pack_padded_sequence(pad_sequence(path).float(), path_len, enforce_sorted=False)
    return path, torch.tensor(path_len), torch.tensor(label), \
           pos_path, torch.tensor(pos_path_len),\
           neg_path, torch.tensor(neg_path_len)


MTU = 1500
class RNN(nn.Module):
    def __init__(self, class_num, out_dim, hidden_size=128, num_layer=3,bidictional=True):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=MTU*2+1,embedding_dim=128,padding_idx=MTU)
        self.rnn = nn.GRU(
            input_size=128, hidden_size=hidden_size, num_layers=num_layer,
            dropout=0.2, bidirectional=bidictional
        )
        out_len = hidden_size*num_layer*2 if bidictional else hidden_size*num_layer
        self.dnn = nn.Sequential(
            nn.BatchNorm1d(num_features=out_len),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_len, out_dim),
        )
        self.cls = nn.Sequential(
            nn.BatchNorm1d(num_features=out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, class_num),
            nn.Softmax(dim=-1)
        )

    def get_embedding(self, path):
        ps = torch.clamp((path[:, :, 1] * path[:, :, -1]).long(), min=-MTU, max=MTU)
        ps_emb = self.embedding(ps + MTU)
        return ps_emb
        # return torch.concat([ps_emb, path], dim=2)

    def forward_from_embedding(self, path_emb, path_len):
        path = pack_padded_sequence(
            path_emb, path_len,
            batch_first=True, enforce_sorted=False
        )
        output, hn = self.rnn(path)
        # hn_ = hn[-1]
        hn_ = torch.cat([hn[i] for i in range(hn.shape[0])], dim=-1)
        out = self.dnn(hn_)
        p = self.sample_prob(out)
        return p

    def sample_forward(self, path, path_len):
        path_emb = self.get_embedding(path)
        path = pack_padded_sequence(
            path_emb, path_len,
            batch_first=True, enforce_sorted=False
        )
        output, hn = self.rnn(path)
        # hn_ = hn[-1]
        hn_ = torch.cat([hn[i] for i in range(hn.shape[0])], dim=-1)
        out = self.dnn(hn_)
        return out

    def sample_prob(self, out):
        p = self.cls(out)
        return p

    def forward(self, path, path_len):
        out = self.sample_forward(path, path_len)
        p = self.sample_prob(out)
        return p

    def transfer_features(self, path, path_len):
        path_emb = self.get_embedding(path)
        path = pack_padded_sequence(
            path_emb, path_len,
            batch_first=True, enforce_sorted=False
        )
        output, hn = self.rnn(path)
        # hn_ = hn[-1]
        hn_ = torch.cat([hn[i] for i in range(hn.shape[0])], dim=-1)
        return hn_




from collections import Counter
from tqdm import tqdm
def train(model, iterator, optimizer, train_type, criterion, metrics, clip, device):
    model.train()
    epoch_loss, epoch_acc = Counter(), 0.0
    pbar = tqdm(iterator, desc=f'Epoch: {epoch + 1}', unit='batch', unit_scale=True)
    for i, data in enumerate(pbar):
        optimizer.zero_grad()
        if train_type == 'cls':
            path, path_len, label = data
            path, label = path.to(device), label.to(device)
            p = model(path,path_len)
            loss = criterion(p, label)
            acc = metrics(p, label)
        elif train_type == 'contr':
            path, path_len, label, pos_path, pos_path_len, neg_path, neg_path_len = data
            path, label = path.to(device), label.to(device)
            pos_path, neg_path = pos_path.to(device), neg_path.to(device)
            out = model.sample_forward(path, path_len)
            out_pos = model.sample_forward(pos_path, pos_path_len)
            out_neg = model.sample_forward(neg_path, neg_path_len)
            loss = criterion(out, out_pos, out_neg)
        elif train_type == 'multi':
            path, path_len, label, pos_path, pos_path_len, neg_path, neg_path_len = data
            path, label = path.to(device), label.to(device)
            pos_path, neg_path = pos_path.to(device), neg_path.to(device)
            out = model.sample_forward(path,path_len)
            p = model.sample_prob(out)
            out_pos = model.sample_forward(pos_path,pos_path_len)
            out_neg = model.sample_forward(neg_path,neg_path_len)
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
            epoch_loss = {key: epoch_loss[key] / len(iterator) for key in epoch_loss.keys()}
            if train_type == 'cls' or train_type == 'multi':
                epoch_acc = epoch_acc / len(iterator)
                pbar.set_postfix(ordered_dict=epoch_loss, acc=f'{epoch_acc:.4f}')
            else:
                pbar.set_postfix(ordered_dict=epoch_loss)

    if train_type == 'cls' or train_type == 'multi':
        return epoch_loss, epoch_acc
    else:
        return epoch_loss



def evaluate(model, iterator, train_type, criterion, metrics, device):
    model.eval()
    epoch_loss, epoch_acc = Counter(), 0.0
    with torch.no_grad():
        for i, data in enumerate(iterator):
            if train_type == 'cls':
                path, path_len, label = data
                path, label = path.to(device), label.to(device)
                p = model(path, path_len)
                loss = criterion(p, label)
                acc = metrics(p, label)
            elif train_type == 'contr':
                path, path_len, label, pos_path, pos_path_len, neg_path, neg_path_len = data
                path, label = path.to(device), label.to(device)
                pos_path, neg_path = pos_path.to(device), neg_path.to(device)
                out = model.sample_forward(path, path_len)
                out_pos = model.sample_forward(pos_path, pos_path_len)
                out_neg = model.sample_forward(neg_path, neg_path_len)
                loss = criterion(out, out_pos, out_neg)
            elif train_type == 'multi':
                path, path_len, label, pos_path, pos_path_len, neg_path, neg_path_len = data
                path, label = path.to(device), label.to(device)
                pos_path, neg_path = pos_path.to(device), neg_path.to(device)
                out = model.sample_forward(path, path_len)
                p = model.sample_prob(out)
                out_pos = model.sample_forward(pos_path, pos_path_len)
                out_neg = model.sample_forward(neg_path, neg_path_len)
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
    loss_triplet = F.triplet_margin_loss(
        anchor=out, positive=out_pos, negative=out_neg,
    )
    # loss_triplet = F.triplet_margin_with_distance_loss(
    #     anchor=out, positive=out_pos, negative=out_neg,
    #     distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
    # )
    return {'loss':loss_triplet}
def multi_loss(p, label, out, out_pos, out_neg):
    loss_c = F.cross_entropy(p,label)
    loss_triplet = F.triplet_margin_loss(anchor=out, positive=out_pos, negative=out_neg,)
    # loss_triplet = F.triplet_margin_with_distance_loss(
    #     anchor=out, positive=out_pos, negative=out_neg,
    #     distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)
    # )
    loss = loss_c + loss_triplet
    return {'loss':loss, 'cls':loss_c, 'triplet':loss_triplet}


def accuracy(pred,label):
    return (pred.argmax(1) == label).float().mean()



from sklearn.model_selection import train_test_split
from datetime import datetime
import time
import argparse

# DEBUG = True
# dataset_name = 'sample'

DEBUG = False
dataset_name = '2100'

test_size = 0.1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--minepoch', type=int, default=25)
    parser.add_argument('--maxwait', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--outdim', type=int, default=16)
    parser.add_argument('--modeltype', type=str, default='RNN')
    parser.add_argument('--traintype', type=str, default='multi')

    args = parser.parse_args()
    BATCH_SIZE = args.batch
    minepoch = args.minepoch
    max_wait = args.maxwait
    hidden_size = args.hidden
    layer_num = args.layers
    out_dim = args.outdim
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

    if train_type == 'cls':
        collate_fn = collate_fn_cls
    elif train_type == 'contr' or train_type == 'multi':
        collate_fn = collate_fn_contr
    else:
        raise Exception(f'Error train type: {train_type}')

    trainset = AppDataset(df=df_train, label2idx=label2idx, train_type=train_type)
    testset = AppDataset(df=df_test, label2idx=label2idx, train_type=train_type)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    if model_type == 'RNN':
        model = RNN(class_num=len(labels), out_dim=out_dim, num_layer=layer_num,hidden_size=hidden_size).to(device)
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
    time.sleep(1)

    N_EPOCHS = 1000
    CLIP = 1
    best_valid_loss = float('inf')

    model_name = f"{train_type}PS-{model_type}"
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
