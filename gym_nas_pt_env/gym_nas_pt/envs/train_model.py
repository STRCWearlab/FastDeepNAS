import torch
import numpy as np
import pandas as pd
from torch import nn
import csv
from collections import deque

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from nas_utils import *
else:
    # uses current package visibility
    from .nas_utils import *

import sklearn.metrics as metrics
from datetime import datetime



def train(model, X_train, y_train, X_val, y_val, batch_size=1000, epochs=15, verbosity='silent', early_stop=False,
          lr_schedule=False, logfile='log', return_avg=5, return_struct=False):
    train_stats = np.unique(y_train, return_counts=True)[1]
    val_stats = np.unique(y_val, return_counts=True)[1]

    if verbosity == 'Full':
        print('Training set statistics:')
        print(len(train_stats), 'classes with distribution', train_stats)
        print('Validation set statistics:')
        print(len(val_stats), 'classes with distribution', val_stats)

    weights = torch.tensor([max(train_stats) / i for i in train_stats], dtype=torch.float).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-8, amsgrad=True)

    criterion = nn.CrossEntropyLoss(weight=weights)
    val_criterion = nn.CrossEntropyLoss()

    start = datetime.now()

    if early_stop:
        early_stopping = EarlyStopping(patience=early_stop, verbose=False)

    if lr_schedule == 'step':
        lr_step = 100
        scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                    lr_step)  # Learning rate scheduler to reduce LR every 100 epochs

    if return_avg:
        criteria = deque([0], maxlen=return_avg)
    else:
        criteria = []

    with open('{}.csv'.format(logfile), 'w', newline='') as csvfile:
        for e in range(epochs):

            train_losses = []
            model.train()
            for batch in iterate_minibatches(X_train, y_train, batch_size, num_batches=None):
                opt.zero_grad()
                x, y = batch

                inputs = torch.from_numpy(x).cuda()
                targets = torch.from_numpy(y).cuda()

                output = model(inputs)

                loss = criterion(output, targets.long())

                loss.backward()
                opt.step()

                train_losses.append(loss.item())
            val_losses = []
            model.eval()  # Setup network for evaluation

            top_classes = []
            targets_cumulative = []

            with torch.no_grad():
                for batch in iterate_minibatches(X_val, y_val, batch_size, num_batches=None):
                    x, y = batch
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    targets_cumulative.extend(y)

                    inputs, targets = inputs.cuda(), targets.cuda()

                    output = model(inputs)
                    val_loss = val_criterion(output, targets.long())
                    val_losses.append(val_loss.item())

                    top_p, top_class = output.topk(1, dim=1)
                    top_classes.extend([top_class.item() for top_class in top_class.cpu()])

            equals = [top_classes[i] == target for i, target in enumerate(targets_cumulative)]
            accuracy = np.mean(equals)

            f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
            f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
            stopping_metric = (f1score + f1macro + accuracy) - np.mean(val_losses)

            if verbosity == 'Full':
                print(
                    'Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Acc: {:.2f}, f1: {:.2f}, M f1: {:.2f}, '
                    'M: {:.4f}'.format(
                        e + 1, epochs, np.mean(train_losses), np.mean(val_losses), accuracy, f1score, f1macro,
                        stopping_metric))

            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([np.mean(train_losses), np.mean(val_losses), accuracy, f1score, f1macro])

            if return_avg:
                criteria.append(stopping_metric)
            else:
                struct = None
                crit = [f1score, np.mean(val_losses), f1macro, accuracy, np.mean(train_losses)]
                if return_struct:
                    crit.append(struct)
                criteria.append(crit)

            if early_stop:
                early_stopping((-stopping_metric), model)
                if early_stopping.early_stop:
                    break
            if lr_schedule:
                scheduler.step()


    end = datetime.now()

    time = (end - start).total_seconds()

    if return_avg:
        crit_stdev = np.std(criteria)
        crit_mean = np.mean(criteria)

        return crit_mean, crit_stdev, time
    else:
        return criteria, None, time


def train_causal(model, X_train, y_train, X_val, y_val, stride, batch_size=1000, epochs=15, verbosity='Full', early_stop=False,
                 lr_schedule=False, logfile='log', return_avg=5, return_struct=False):

    train_stats = np.unique([a for y in y_train for a in y], return_counts=True)[1]
    val_stats = np.unique([a for y in y_val for a in y], return_counts=True)[1]

    if verbosity == 'Full':
        print('Training set statistics:')
        print(len(train_stats), 'classes with distribution', train_stats)
        print('Validation set statistics:')
        print(len(val_stats), 'classes with distribution', val_stats)

    weights = torch.tensor([max(train_stats) / i for i in train_stats], dtype=torch.float).cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-8, amsgrad=True)

    criterion = nn.CrossEntropyLoss(weight=weights)
    val_criterion = nn.CrossEntropyLoss()

    start = datetime.now()

    if early_stop:
        early_stopping = EarlyStopping(patience=early_stop, verbose=False)

    if lr_schedule == 'step':
        lr_step = 100
        scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                    lr_step)  # Learning rate scheduler to reduce LR every 100 epochs

    if return_avg:
        criteria = deque([0], maxlen=return_avg)
    else:
        criteria = []

    with open('{}.csv'.format(logfile), 'w', newline='') as csvfile:
        for e in range(epochs):

            train_losses = []
            model.train()

            for batch in iterate_minibatches_2D(X_train, y_train, batch_size, stride, num_batches=2,
                                                batchlen=5, drop_last=True):
                opt.zero_grad()

                x, y, pos = batch

                inputs = torch.from_numpy(x).cuda()
                targets = torch.from_numpy(y).cuda()

                if pos == 0:
                    h = model.init_hidden(batch_size)

                output, h = model(inputs, h)

                loss = criterion(output, targets.long())

                loss.backward()
                opt.step()

                train_losses.append(loss.item())


            val_losses = []
            model.eval()  # Setup network for evaluation

            top_classes = []
            targets_cumulative = []

            with torch.no_grad():
                for batch in iterate_minibatches_2D(X_val, y_val, batch_size, stride, num_batches=2,
                                                    batchlen=5, drop_last=True):
                    x, y, pos = batch

                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                    targets_cumulative.extend([y for y in y])

                    if pos == 0:
                        val_h = model.init_hidden(batch_size)

                    inputs, targets = inputs.cuda(), targets.cuda()
                    output, val_h = model(inputs, val_h)

                    val_loss = val_criterion(output, targets.long())
                    val_losses.append(val_loss.item())

                    top_p, top_class = output.topk(1, dim=1)
                    top_classes.extend([top_class.item() for top_class in top_class.cpu()])


            equals = [top_classes[i] == target for i, target in enumerate(targets_cumulative)]
            accuracy = np.mean(equals)

            f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
            f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
            stopping_metric = (f1score + f1macro + accuracy) - np.mean(val_losses)

            if verbosity == 'Full':
                print(
                    'Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Acc: {:.2f}, f1: {:.2f}, M f1: {:.2f}, '
                    'M: {:.4f}'.format(
                        e + 1, epochs, np.mean(train_losses), np.mean(val_losses), accuracy, f1score, f1macro,
                        stopping_metric))

            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([np.mean(train_losses), np.mean(val_losses), accuracy, f1score, f1macro])

            if return_avg:
                criteria.append(stopping_metric)
            else:
                struct = None
                crit = [f1score, np.mean(val_losses), f1macro, accuracy, np.mean(train_losses)]
                if return_struct:
                    crit.append(struct)
                criteria.append(crit)

            if early_stop:
                early_stopping((-stopping_metric), model)
                if early_stopping.early_stop:
                    break
            if lr_schedule:
                scheduler.step()



    end = datetime.now()

    time = (end - start).total_seconds()

    if return_avg:
        crit_stdev = np.std(criteria)
        crit_mean = np.mean(criteria)

        return crit_mean, crit_stdev, time
    else:
        return criteria, None, time

if __name__ == '__main__':
    pass