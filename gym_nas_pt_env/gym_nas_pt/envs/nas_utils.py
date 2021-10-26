import _pickle as cp
import csv
import glob
import os
import time
from collections import Counter  # Counter counts the number of occurrences of each item
from itertools import tee, count

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def init_weights(m):
    """Initialize weights for LSTM and Conv1d layers."""
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)


def iterate_minibatches(inputs, targets, batchsize, shuffle=True, num_batches=None):
    """Iterate through a (segmented) dataset, yielding minibatches of of (data,label) pairs.

    Yields batchsize (data, label) pairs drawn from inputs and targets each iteration. Drops the last few windows if
    len(inputs) is not a multiple of batchsize.

    Optional arguments:
    shuffle -- If True, shuffle the batches
    num_batches -- If not None, terminate after num_batches batches
    """
    batch = lambda j: [x for x in range(j * batchsize, (j + 1) * batchsize)]

    batches = [i for i in range(int(len(inputs) / batchsize))]

    if shuffle:
        np.random.shuffle(batches)
        for i in batches[:num_batches]:
            yield np.array([inputs[i] for i in batch(i)]), np.array([targets[i] for i in batch(i)])

    else:
        for i in batches[:num_batches]:
            yield np.array([inputs[i] for i in batch(i)]), np.array([targets[i] for i in batch(i)])


def plot_data(logname='log.csv', save_fig=False):
    """Plot training and validation statistics from a csv file."""
    train_loss_plot = []
    val_loss_plot = []
    acc_plot = []
    f1_plot = []
    f1_macro = []

    with open(logname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            train_loss_plot.append(float(row[0]))
            val_loss_plot.append(float(row[1]))
            acc_plot.append(float(row[2]))
            f1_plot.append(float(row[3]))
            f1_macro.append(float(row[4]))

    if save_fig:
        try:
            os.makedirs('Results/{}'.format(save_fig))
        except FileExistsError:
            pass

    plt.figure(1)
    plt.title('Training loss')
    plt.ylabel('Categorical cross entropy')
    plt.xlabel('Epoch')
    plt.plot(train_loss_plot)
    if save_fig:
        plt.savefig('Results/{}/Train_loss_{}.png'.format(save_fig, time.time()))

    plt.figure(2)
    plt.title('Validation loss')
    plt.ylabel('Categorical cross entropy')
    plt.xlabel('Epoch')
    plt.plot(val_loss_plot)
    if save_fig:
        plt.savefig('Results/{}/Val_loss_{}.png'.format(save_fig, time.time()))

    plt.figure(3)
    plt.title('Validation Accuracy')
    plt.ylabel('Weighted accuracy')
    plt.xlabel('Epoch')
    plt.plot(acc_plot)
    if save_fig:
        plt.savefig('Results/{}/Val_acc_{}.png'.format(save_fig, time.time()))

    plt.figure(4)
    plt.title('Validation f1 score')
    plt.ylabel('f1 score')
    plt.xlabel('Epoch')
    plt.plot(f1_plot, label='Weighted')
    plt.plot(f1_macro, label='Macro')
    plt.legend()
    if save_fig:
        plt.savefig('Results/{}/Val_f1_{}.png'.format(save_fig, time.time()))

    if not save_fig:
        plt.show()


def load_data(name, len_seq, stride, keep_seperate=False):
    """Load pickled data from multiple files on disk."""
    Xs = []
    ys = []

    ## Use glob module and wildcard to build a list of files to load from data directory
    path = os.path.expanduser("~/NAS/data/{}_data_*".format(name))
    data = glob.glob(path)

    for file in data:
        X, y = load_dataset(file)
        X, y = slide(X, y, len_seq, stride, save=False)

        if keep_seperate:
            Xs.append(X)
            ys.append(y)
        else:
            Xs.extend(X)
            ys.extend(y)

    return Xs, ys


def load_dataset(filename):
    """Load pickled data from a file on disk."""
    with open(filename, 'rb') as f:
        data = cp.load(f)

    X, y = data

    print('Got {} samples from {}'.format(X.shape, filename))

    X = X.astype(np.float32)
    y = y.astype(np.uint8)

    return X, y


def slide(data_x, data_y, ws, ss, save=False):
    """Segment a stream of data, label pairs with a sliding window."""
    x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    y = np.asarray([i[-1] for i in sliding_window(data_y, ws, ss)]).astype(np.uint8)

    if save:
        with open('data_slid', 'wb') as f:
            cp.dump((x, y), f, protocol=4)

    else:
        return x, y


def iterate_minibatches_2D(inputs, targets, batchsize=1000, stride=1, num_batches=20, batchlen=50, drop_last=True,
                           shuffle=True):
    """Iterate through a dataset which has been segmented with a sliding window and kept in separate arrays where
    necessary to preserve continuity in time.

    Args:
        inputs (array): Dataset sensor channels, a stacked array of runs, after a sliding window has been applied.

        targets (array): Dataset labels, a stacked array of labels corresponding to the windows in inputs.

        batchsize (int): Number of windows in each batch.

        stride (int): Size of sliding window step.

        num_batches (int): Number of metabatches to return before finishing the epoch.
                Default: 10
        batchlen (int): Number of contiguous windows per batch.
                Default: 50
        drop_last (bool): Whether to drop the last incomplete batch when dataset does not divide neatly by batchsize.
                Default: True
        shuffle (bool): Determines whether to shuffle the batches or iterate through sequentially.
                Default: True
    """
    window_size = len(inputs[0][0])
    assert (
            window_size / stride).is_integer(), 'in order to generate sequential batches, the sliding window length ' \
                                                'must be divisible by the step. '

    starts = [[x for x in range(0, len(i) - int(((batchlen * window_size) + 1) / stride))] for i in inputs]

    for i in range(1, len(starts)):
        starts[i] = [x + 1 + starts[i - 1][-1] + int(((batchlen * window_size) + 1) / stride) for x in starts[i]]

    starts = [val for sublist in starts for val in sublist]
    inputs = [val for sublist in inputs for val in sublist]
    targets = [val for sublist in targets for val in sublist]

    step = lambda x: [int(x + i * window_size / stride) for i in range(batchlen)]

    if shuffle:
        np.random.shuffle(starts)

    batches = np.empty((batchsize, batchlen), dtype=np.int32)

    if num_batches != -1:
        num_batches = int(num_batches * batchsize)  # Convert num_batches to number of metabatches.
        if num_batches > len(starts):
            num_batches = -1

    for i, start in enumerate(starts[0:num_batches]):

        batch = np.array([i for i in step(start)], dtype=np.int32)

        batches[i % batchsize] = batch

        if i % batchsize == batchsize - 1:

            batches = batches.transpose()

            for pos, batch in enumerate(batches):
                yield np.array([inputs[i] for i in batch]), np.array([targets[i] for i in batch]), pos
                batches = np.empty((batchsize, batchlen), dtype=np.int32)

        if drop_last == False and i == len(starts) and i % batchsize != 0:

            batches = batches[0:i % batchsize]
            batches = batches.transpose()
            for pos, batch in enumerate(batches):
                yield np.array([inputs[i] for i in batch]), np.array([targets[i] for i in batch]), pos


def iterate_minibatches_test(inputs, targets, window_size, stride):
    """Iterate through the testing set in one go, keeping contiguity in time."""
    assert (
            window_size / stride).is_integer(), 'in order to generate sequential batches, the sliding window length ' \
                                                'must be divisible by the step. '

    starts = [[(x, int(np.floor(len(i) / window_size))) for x in range(0, window_size)] for i in inputs]

    # for i in range(1,len(starts)):
    # 	starts[i] = [(x+1+len(inputs[i-1]),j) for x,j in starts[i]]

    starts = [sublist for sublist in starts]
    inputs = [sublist for sublist in inputs]
    targets = [sublist for sublist in targets]

    step = lambda x, j: [int(x + i * window_size / stride) for i in range(j)]

    for i in range(len(inputs)):

        for start in starts[i][0:window_size]:
            start, batchlen = start
            batches = np.array([np.array([i for i in step(start[0], start[1])]) for start in starts[i][0:window_size]])

        batches = batches.transpose()

        for pos, batch in enumerate(batches):
            yield np.array([inputs[i][j] for j in batch]), np.array([targets[i][j] for j in batch]), pos


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# from http://www.johnvinyard.com/blog/?p=268
from numpy.lib.stride_tricks import as_strided as ast


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.
    Parameters
        shape - an int, or a tuple of ints
    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions
    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.
    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)

    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape), len(ws), len(ss)]
    if 1 != len(set(ls)):
        raise ValueError( \
            'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError( \
            'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape), str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a, shape=newshape, strides=newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #     dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def int_to_one_hot(integer, size):
    one_hot = np.zeros(size)  # Convert layer to a one-hot vector
    one_hot[integer] = 1
    return one_hot


def one_hot_to_int(one_hot):
    integer = int(np.where(one_hot == 1)[0])
    return integer


def int_to_binary(integer, bitwidth):
    binstring = bin(integer)[2:]
    if len(binstring) > bitwidth:
        raise ValueError('Integer is too large to be represented in {} bits, choose a different encoding' \
                         .format(bitwidth))
    elif len(binstring) == bitwidth:
        return [int(i) for i in binstring]
    else:
        zeros = [0 for _ in range(bitwidth)]
        binstring = [int(i) for i in binstring]
        binstring.reverse()
        for i, bit in enumerate(binstring):
            zeros[i] = bit
        zeros.reverse()
        return zeros


def uniquify(seq, suffs=count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).

    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k, v in Counter(seq).items() if v > 1]  # so we have: ['name', 'zip']
    # suffix generator dict - e.g., {'name': <my_gen>, 'zip': <my_gen>}
    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            # s was unique
            continue
        else:
            seq[idx] += '{}'.format(suffix)
    suffs = count(1)
