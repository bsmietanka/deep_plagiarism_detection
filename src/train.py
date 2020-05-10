import argparse
import json
from os.path import join
from datetime import datetime
from shutil import copy

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch import nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from datasets import Dataset
from models import SiameseModel
from utils.r_tokenizer import tokenize, vocab_size
from metrics import Accuracy, ConfusionMatrix

start_time = datetime.now()

runs_dir = "runs"
leaderboards = join(runs_dir, 'leaderboards.csv')
save_path = join(runs_dir, str(start_time))
writer = SummaryWriter(save_path)

# TODO: combine multiple datasets and use class balancing

# losses = {
#     'ce': lambda **kwargs: MultiClassLossWrapper(nn.CrossEntropyLoss, **kwargs),
#     'bce' : nn.BCELoss,
#     'focal': FocalLoss
# }

# metrics = {
#     'accuracy': Accuracy,
#     'confusion_matrix': ConfusionMatrix
# }

def write2file(path: str, text: str):
    with open(join(save_path, path), 'w') as f:
        f.write(text.strip() + "\n")

# TODO: learning rate scheduler?
# TODO: optimizer in config

def load_model(model_path: str) -> nn.Module:
    if model_path == "best":
        df: pd.DataFrame = pd.read_csv(leaderboards, names=["run", "val_score", "test_score"])
        idxmax = df['val_score'].idxmax()
        model_path = join(runs_dir, df.loc[idxmax, 'run'], 'model.pth')
    if model_path == "last":
        with open(leaderboards, 'r') as f:
            last_line:str = f.readlines()[-1]
            model_path = join(runs_dir, last_line.split(',')[0], 'model.pth')
    print("Loading model at:", model_path)
    write2file('checkpoint_from.txt', model_path)
    return torch.load(model_path)

def train(**params):

    best_acc = 0
    test_acc = 0
    batch = params.get('batch', 50)
    epochs = params.get('epochs', 100)

    device = params.get('device', 'cuda')
    print("Running on:", device)

    data_params = params["dataset_params"]

    # TODO: split the dataset somehow
    # TODO: combine multiple datasets
    dataset = Dataset(**data_params)

    indexes = list(range(len(dataset)))
    print(len(indexes))
    train_ind, val_ind = train_test_split(indexes, test_size=0.2)

    train_sampler = SubsetRandomSampler(train_ind)
    val_sampler = SubsetRandomSampler(val_ind)
    test_sampler = None # TODO when trainning starts working

    def pack_sequences(sequences):
        seq_lengths = [len(seq) for seq in sequences]
        padded_seq = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed_seq = nn.utils.rnn.pack_padded_sequence(padded_seq, seq_lengths,
                                                       batch_first=True, enforce_sorted=False)
        return packed_seq

    def manual_batching(batch):
        fun1, fun2, target, similarity = zip(*batch)
        target = torch.tensor(target).float()
        # similarity = torch.tensor(similarity)
        return pack_sequences(fun1), pack_sequences(fun2), target, similarity

    trainloader = DataLoader(dataset, num_workers=8, batch_size=batch, sampler=train_sampler,
                             collate_fn=manual_batching)
    valloader = DataLoader(dataset, num_workers=8, batch_size=batch, sampler=val_sampler,
                           collate_fn=manual_batching)
    # testloader = DataLoader(test_dataset, num_workers=8, batch_size=batch,
    #                         collate_fn=manual_batching, shuffle=False)

    train_log_interval = len(trainloader)
    val_log_interval = len(valloader)

    # TODO: Model params in config
    model = SiameseModel(vocab_size, 32, 128, 1, 1)
    if 'load_model' in params:
        model.load_state_dict(load_model(params['load_model']))

    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), **params['optimizer_params'])
    # TODO: check lr_scheduler docs
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    no_improvement = 0

    metric = Accuracy()
    metric2 = ConfusionMatrix()

    try:
        for epoch in range(1, epochs + 1):
            print(f'EPOCH #{epoch}')

            print('Training')
            model.train()
            running_loss = 0.
            
            for i, batch in enumerate(tqdm(trainloader), 1):

                optimizer.zero_grad()

                fun1_sequences = batch[0].to(device)
                fun2_sequences = batch[1].to(device)
                labels = batch[2].to(device)
                similarity = batch[3]

                res = model.forward(fun1_sequences, fun2_sequences)

                loss = criterion(res, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % train_log_interval == 0:
                    writer.add_scalar('train loss', running_loss / train_log_interval, i + epoch * len(trainloader))
                    running_loss = 0.0

            print('Validation')
            model.eval()
            running_loss = 0.
            with torch.no_grad():
                for i, batch in enumerate(tqdm(valloader), 1):

                     # TODO: batch handling
                    fun1_sequences = batch[0].to(device)
                    fun2_sequences = batch[1].to(device)
                    labels = batch[2].to(device)
                    similarity = batch[3]

                    res = model.forward(fun1_sequences, fun2_sequences)
                    loss = criterion(res, labels)

                    # TODO: replace with metric, like accuracy, f1 score etc.
                    metric(res, labels)
                    metric2(res, labels)

                    running_loss += loss.item()

                    if i % val_log_interval == 0:
                        writer.add_scalar('val_loss', running_loss / val_log_interval, i + epoch * len(valloader))
                        running_loss = 0.0

            print("Confussion matrix:", metric2)
            metric2.reset()

            print('Validation metric:', metric)
            writer.add_scalar(f'validation {metric.name}', metric.get(), epoch)

            if metric.get() > best_acc: # TODO: replace with metric comparison? How would it work with confusion matrix, not very good solution
                no_improvement = 0
                torch.save(model.state_dict(), join(save_path, 'model.pth'))
                write2file('best_epoch.txt', f'Best epoch: {epoch}')
                best_acc = metric.get()
            else:
                no_improvement += 1
                if 'early_stopping' in params and no_improvement >= params['early_stopping']:
                    break
            metric.reset()
            
            # scheduler.step(metric.get())

        # load best model
        # model.load_state_dict(torch.load(join(save_path, 'model.pth')))
        # print('Testing')
        # metric.reset()
        # model.eval()
        # running_loss = 0.
        # with torch.no_grad():
        #     for i, batch in enumerate(tqdm(testloader), 1):
        #         fun1_sequences = batch[0].to(device)
        #         fun2_sequences = batch[1].to(device)
        #         labels = batch[2].to(device)
        #         similarity = batch[3]

        #         outputs = model.forward(fun1_sequences, fun2_sequences)
        #         metric(outputs, labels)

        # test_acc = metric.get()

    finally:
        with open(leaderboards, 'a') as f:
            f.write(f'{start_time},{best_acc},{test_acc}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for multilabel face attributes classification model training'
        )
    parser.add_argument('--config_path', '-c', type=str, default='config.json',
        help='Path to config file with parameters, look at config.json')
    args = parser.parse_args()

    # with open('config.json', 'r') as f:

    with open(args.config_path, 'r') as f:
        params = json.load(f)

    copy(args.config_path, save_path)

    train(**params)
