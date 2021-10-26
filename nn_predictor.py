import torch
from torch import nn
from utils import iterate_minibatches
from scipy.stats import spearmanr
import numpy as np


class Predictor(nn.Module):

    def __init__(self, input_channels, n_units=32):
        super().__init__()

        self.input = nn.Linear(input_channels, n_units)

        self.linear1 = nn.Linear(n_units, n_units)

        self.linear2 = nn.Linear(n_units, 1)

        self.output = nn.Sigmoid()

    def forward(self, x):

        x = self.input(x)

        x = self.linear1(x)

        x = self.linear2(x)

        x = self.output(x)

        return x


class BranchedPredictor(nn.Module):

    def __init__(self, input_channels, channel_length, n_params, n_units=32, struct=False):

        self.struct = struct

        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, n_units, (3,))

        self.conv2 = nn.Conv1d(10, n_units, (3,))

        self.flatten = nn.Flatten(1)

        if struct:
            flatten_dim = n_units * (channel_length-2) + n_units * 5 + n_params
        else:
            flatten_dim = n_units * (channel_length-2) + n_params

        self.linear1 = nn.Linear(flatten_dim, n_units)

        self.linear2 = nn.Linear(n_units, 1)

        self.output = nn.Sigmoid()

    def forward(self, metrics, struct, extra_params):

        metrics = self.conv1(metrics)

        if self.struct:
            struct = self.conv2(struct)

            x = torch.cat((self.flatten(metrics), self.flatten(struct), extra_params), dim=1)
        else:
            x = torch.cat((self.flatten(metrics), extra_params), dim=1)

        x = self.linear1(x)

        x = self.linear2(x)

        x = self.output(x)

        return x


def init_weights(m):
    if type(m) == nn.Linear:
        m.reset_parameters()


def weighted_mse_loss(pred, targets, weights):
    return ((weights * (pred - targets)) ** 2).mean()


def train_predictor(predictor, X_train, y_train, n_epochs=10, weighted=True, branched=False, eps=None, n_channels=4,
                    struct=False):
    predictor.apply(init_weights)
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 100)
    structures = False

    for epoch in range(n_epochs):

        predictor.train()

        train_losses = []
        train_corrs = []

        for batch in iterate_minibatches(X_train, y_train, 50):
            opt.zero_grad()
            x, y = batch

            if branched:
                inputs = np.array([np.array(x[:eps * n_channels]).reshape(n_channels, eps) for x in x])
                if struct:
                    structures = np.array([np.array(x[eps * n_channels:-2]).reshape(10, 7) for x in x])
                    structures = torch.from_numpy(structures).cuda()
                params = np.array([np.array(x[-2:]) for x in x])
                inputs, params, targets = torch.from_numpy(inputs).cuda(), torch.from_numpy(params).cuda(), \
                                          torch.from_numpy(y).cuda()

                output = predictor(inputs, structures, params)
            else:
                inputs, targets = torch.from_numpy(x).cuda().squeeze(1), torch.from_numpy(y).cuda()

                output = predictor(inputs)
            #
            # loss = -pearsonr(output.squeeze(1), targets)
            loss = weighted_mse_loss(output.squeeze(1), targets, targets)
            loss.backward()
            opt.step()
            train_corr = spearmanr(output.detach().cpu(), targets.cpu())[0]
        
            train_losses.append(loss.item())
            train_corrs.append(train_corr)
            
#             scheduler.step(train_corr)

        print('\r {} Loss: {:.5f}, Corr: {:.2f}'.format(epoch, np.mean(train_losses), np.mean(train_corrs)), end='')
