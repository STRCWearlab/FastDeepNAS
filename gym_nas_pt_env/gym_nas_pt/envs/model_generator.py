if __package__ is None or __package__ == '':
    # uses current directory visibility
    from NSC import *
    from train_model import *
    from nas_utils import *
else:
    # uses current package visibility
    from .NSC import *
    from .train_model import *
    from .nas_utils import *
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.functional import pad


class ModelSpec:

    def __init__(self, layers):

        """ Builds a model encoding from a list of NSC Dataframes """
        self.hashable = tuple([layer.tuple for layer in layers])
        self.encoding = pd.DataFrame()
        self.n_layers = len(layers)
        for layer in layers:
            self.encoding = self.encoding.append(layer.repr)  # Add layers
        self.edges = self.get_edges(self.encoding)
        self.loose_ends = self.get_loose_ends(self.edges)
        self.nodes = self.get_nodes(self.encoding)

    def write(self, n_channels, window_size, batch_size, classifier='Linear'):
        model = NASNet(self.encoding, n_classes=18, n_channels=n_channels,
                       window_size=window_size, batch_size=batch_size, classifier=classifier)  # Write model
        return model

    def get_loose_ends(self, edges):
        loose_ends = []
        for layer, succ in enumerate(edges.T[1:-1]):
            if len(np.where(succ == 1)[0]) == 0:
                loose_ends.append(layer + 1)
        return loose_ends

    def get_edges(self, encoding):

        """ Generate graph edges representing connections between layers,
            according to predecessors defined in the arch """

        edges = np.zeros((self.n_layers, self.n_layers), dtype=int)

        for i in range(1, self.n_layers - 1):

            edges[encoding['Pred1'][i]][i] = 1

            if encoding['Type'][i] == 'Concatenation':
                edges[encoding['Pred2'][i]][i] = 1

            if encoding['Type'][i] == 'Elementwise Addition':
                edges[encoding['Pred2'][i]][i] = 1

        edges = edges.T
        return edges

    def density(self):

        edges = self.edges.T

        # Get the edges connecting to input / terminal layers which we leave out
        for i in range(1, self.n_layers - 1):

            if self.encoding['Pred1'][i] == 0:
                edges[0][i] = 1

        edge_sum = sum(sum(edges))
        node_sum = len(self.nodes)

        if len(self.loose_ends) > 0:
            edge_sum += len(self.loose_ends)
            node_sum += 1
        else:
            edge_sum += 1

        return edge_sum / node_sum

    def get_nodes(self, encoding):

        return list(encoding['Type'])


class NASNet(nn.Module):

    def __init__(self, architecture, n_channels=12,
                 n_classes=18, window_size=60, batch_size=BATCH_SIZE,
                 classifier='Linear'):

        """ Builds a torch model from a dataframe describing the desired
            architecture in terms of NSC """

        super(NASNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.window_size = window_size
        self.batch_size = batch_size
        self.classifier = classifier

        self.arch = architecture
        self.n_layers = len(self.arch)

        self.operations = self.arch['Type']

        # Sort layers by predecessor so we can construct them rationally
        self.arch['max'] = [max(i, j) for i, j in zip(*(self.arch['Pred1'],
                                                        self.arch['Pred2']))]
        self.arch = self.arch.sort_values('max')

        self.edges = self.get_edges()
        self.loose_ends = self.get_loose_ends()
        self.module_list = self.get_nodes()

    def get_edges(self):

        """ Generate graph edges representing connections between layers,
            according to predecessors defined in the arch """

        edges = np.zeros((self.n_layers, self.n_layers), dtype=int)

        for i in range(1, self.n_layers - 1):

            edges[self.arch['Pred1'][i]][i] = 1

            if self.arch['Type'][i] == 'Concatenation':
                edges[self.arch['Pred2'][i]][i] = 1

            if self.arch['Type'][i] == 'Elementwise Addition':
                edges[self.arch['Pred2'][i]][i] = 1

        edges = edges.T
        return edges

    def get_nodes(self):

        """ Generate graph nodes (operations) representing layers,
            according to arch """

        intermediate_channels = [None for i in range(self.n_layers)]
        intermediate_len = [None for i in range(self.n_layers)]
        intermediate_channels[0] = self.n_channels
        intermediate_len[0] = self.window_size

        modules = nn.ModuleList([])

        for row in self.arch.iterrows():

            layer_type = row[1]['Type']

            if layer_type == 'Convolution':
                kernel_size = row[1]['Kernel_size']
                n_filters = row[1]['N_kernels']
                padding = int(kernel_size / 2)

                input_channels = intermediate_channels[row[1]['Pred1']]
                input_len = intermediate_len[row[1]['Pred1']]

                modules.append(BNConvReLU(input_channels, n_filters, kernel_size, padding=padding))
                intermediate_channels[row[0]] = n_filters
                intermediate_len[row[0]] = (input_len + 2 * padding - (kernel_size - 1))

            if layer_type == 'Max Pooling':
                kernel_size = row[1]['Kernel_size']
                modules.append(nn.MaxPool1d(kernel_size))
                input_channels = intermediate_channels[row[1]['Pred1']]
                input_len = intermediate_len[row[1]['Pred1']]
                intermediate_channels[row[0]] = input_channels
                intermediate_len[row[0]] = int(((input_len - (kernel_size - 1) - 1) / kernel_size) + 1)

            if layer_type == 'Average Pooling':
                kernel_size = row[1]['Kernel_size']
                modules.append(nn.AvgPool1d(kernel_size))
                input_channels = intermediate_channels[row[1]['Pred1']]
                input_len = intermediate_len[row[1]['Pred1']]
                intermediate_channels[row[0]] = input_channels
                intermediate_len[row[0]] = int(((input_len - kernel_size) / kernel_size) + 1)

            if layer_type == 'Identity':
                modules.append(Identity())
                input_channels = intermediate_channels[row[1]['Pred1']]
                input_len = intermediate_len[row[1]['Pred1']]
                intermediate_channels[row[0]] = input_channels
                intermediate_len[row[0]] = input_len

            if layer_type == 'Elementwise Addition':
                modules.append(ElementwiseAdd())
                input_channels = max(intermediate_channels[row[1]['Pred1']], intermediate_channels[row[1]['Pred2']])
                input_len = max(intermediate_len[row[1]['Pred1']], intermediate_len[row[1]['Pred2']])
                intermediate_channels[row[0]] = input_channels
                intermediate_len[row[0]] = input_len

            if layer_type == 'Concatenation':
                modules.append(Cat())
                input_channels = intermediate_channels[row[1]['Pred1']] + intermediate_channels[row[1]['Pred2']]
                input_len = max(intermediate_len[row[1]['Pred1']], intermediate_len[row[1]['Pred2']])
                intermediate_channels[row[0]] = input_channels
                intermediate_len[row[0]] = input_len

        # Add final concatenation if number of loose ends > 1
        if len(self.loose_ends) > 1:
            final_channels = sum(
                [intermediate_channels[i] for i in self.loose_ends])  # Sum channels from all loose ends
            final_len = max([intermediate_len[i] for i in self.loose_ends])  # Sum length from all loose ends
            self.final_cat = Cat()
            intermediate_channels[-1] = final_channels
            intermediate_len[-1] = final_len

        # Add two layer MLP with input dim equal to the flattened dimensions of the concatenated loose ends
        self.intermediate_channels = [i for i in intermediate_channels if i is not None]
        self.intermediate_len = [i for i in intermediate_len if i is not None]

        if self.classifier == 'Linear':
            self.flattened_dim = self.intermediate_channels[-1] * self.intermediate_len[-1]
            self.MLP1 = nn.Linear(self.flattened_dim, 128)
            self.MLP2 = nn.Linear(128, 128)

        elif self.classifier == 'LSTM':
            self.lstm = nn.LSTM(self.intermediate_channels[-1], 128, num_layers=2, batch_first=True)

        self.output = nn.Linear(128, self.n_classes)

        return modules

    def get_loose_ends(self):
        loose_ends = []
        for layer, succ in enumerate(self.edges.T[1:-1]):
            if len(np.where(succ == 1)[0]) == 0:
                loose_ends.append(layer + 1)
        return loose_ends

    def forward(self, x, h=None):

        """ Here we iterate through the operations in index order, and for
        each operation we iterate through the possible predecessors. self.edges stores
        the connections between operations so we check for each pair of op,pred whether
        there is a connection, and if so then we perform the operation on the output of the
        predecessor/s, stored in the intermediate_values list."""
        module = 0

        x = x.reshape(BATCH_SIZE, self.n_channels, self.window_size)

        intermediate_values = [None for i in range(self.n_layers)]
        intermediate_values[0] = x

        done_layers = []

        for layer in self.arch.index:

            # Check for predecessors - implement layer with 1 pred
            if len(np.where(self.edges[layer] == 1)[0]) == 1:
                pred = self.arch['Pred1'][layer]

                x = intermediate_values[pred]
                x = self.module_list[module](x)

                intermediate_values[layer] = x
                module += 1
            # Implement layer with 2 preds
            if len(np.where(self.edges[layer] == 1)[0]) == 2:
                predecessors = (self.arch['Pred1'][layer], self.arch['Pred2'][layer])
                predecessors = tuple(intermediate_values[p] for p in predecessors)
                x = self.module_list[module](predecessors)

                intermediate_values[layer] = x
                module += 1

            # Check if we still need intermediate values. Values are kept if
            # 1) they have successor layers which have not yet been implemented or
            # 2) they are 'loose ends' and must be concatenated into the final layer
            done_layers.append(layer)  # Keep a list of implemented layers

            # Iterate over successors
            for layer, succ in enumerate(self.edges.T):

                successors = np.where(succ == 1)[0]  # Get successor indexes
                layer_done = [s in done_layers for s in successors]  # Check if all successors have been implemented

                # If all successors have been implemented, and layer is not a loose end, delete intermediate output
                if all(layer_done) and layer not in self.loose_ends:
                    intermediate_values[layer] = None

        # Now concatenate any layers (not terminal or input)
        # with no successors (i.e. loose ends)
        # and send to the MLP
        if len(self.loose_ends) > 0:
            x = tuple(intermediate_values[l] for l in self.loose_ends)
            del intermediate_values

        if len(self.loose_ends) > 1:
            x = self.final_cat(x)

        elif len(self.loose_ends) == 1:
            x = x[0]

        else:
            x = x

        if self.classifier == 'LSTM':
            x = x.view(self.batch_size, -1, self.intermediate_channels[-1])
            x, _ = self.lstm(x, h)
            x = x.view(self.batch_size, -1, 128)[:, -1, :]

            x = self.output(x)

            return x, h

        else:

            x = x.reshape(self.batch_size, self.flattened_dim)

            x = self.MLP1(x)
            x = self.MLP2(x)

            x = self.output(x)

            return x

    def init_hidden(self, batch_size):

        weight = next(
            self.parameters()).data  # return a Tensor from self.parameters to use as a base for the initial hidden state.

        ## Generate new tensors of zeros with similar type to weight, but different size.
        hidden = (weight.new_zeros(2, batch_size, 128).cuda(),  # Hidden state
                  weight.new_zeros(2, batch_size, 128).cuda())  # Cell state

        return hidden


class Cat(nn.Module):

    def forward(self, pred):
        return self.safe_cat(pred)

    def safe_cat(self, pred):
        """ Concatenates a tuple of tensors pred by channels
        , with padding if the sequence lengths are mismatched.
        Each element of pred should be of shape (batch_size,*,n_channels,len_sequence). """

        if type(pred) == tuple:

            pad_sizes = [[] for i in pred]
            pred_sizes = [i.size() for i in pred]

            # Check if all lengths are equal, then we don't need to pad
            if all(x[-1] == pred_sizes[0][-1] for x in pred_sizes):
                return torch.cat(pred, axis=1)

            else:
                target_len = max([x[-1] for x in pred_sizes])  # Get max seq_len

                # Get size diff for each tensor
                for i, tens in enumerate(pred):
                    if tens.size()[-1] < target_len:
                        pad_sizes[i].append(int(np.floor((target_len - tens.size()[-1]) / 2)))
                        pad_sizes[i].append(int(np.ceil((target_len - tens.size()[-1]) / 2)))
                    else:
                        pad_sizes[i].extend((0, 0))

                pred_padded = [pad(pred, tuple(pad_sizes[i])) for i, pred in enumerate(pred)]  # Do padding

                return torch.cat(pred_padded, axis=1)

        else:
            return pred


class Identity(nn.Module):

    def forward(self, pred):
        return pred


class ElementwiseAdd(nn.Module):

    def forward(self, pred):
        return self.safe_add(pred)

    def safe_add(self, pred):

        if type(pred) == tuple:

            pad_sizes = [[] for i in pred]
            pred_sizes = [i.size() for i in pred]

            # Check if all lengths and n_channels are equal
            if all(x[ax] == pred_sizes[0][ax] for x in pred_sizes for ax in [-1, -2]):
                return torch.sum(torch.stack(pred, axis=0), axis=0)

            else:
                target_sizes = [max(x[ax] for x in pred_sizes) for ax in [-1, -2]]

                # Get size diff for each tensor
                for k, ax in enumerate([-1, -2]):
                    for i, tens in enumerate(pred):
                        if tens.size()[ax] < target_sizes[k]:
                            pad_sizes[i].append(int(np.floor((target_sizes[k] - tens.size()[ax]) / 2)))
                            pad_sizes[i].append(int(np.ceil((target_sizes[k] - tens.size()[ax]) / 2)))
                        else:
                            pad_sizes[i].extend((0, 0))

                pred_padded = [pad(pred, tuple(pad_sizes[i])) for i, pred in enumerate(pred)]  # Do padding

                return torch.sum(torch.stack(pred_padded, axis=0), axis=0)

        else:
            return pred


class BNConvReLU(nn.Module):

    def __init__(self, input_channels, n_filters, kernel_size, padding):
        """ Generate a Conv-ReLU-BatchNorm block based on specified parameters.
            :input_channels: integer
                number of input channels
            :n_filters: integer
                number of convolutional filters to apply
            :kernel_size: integer
                size of convolutional kernel
            :padding: integer
                size of padding to append to each end of the input sequence"""

        super().__init__()

        self.conv = nn.Conv1d(input_channels, n_filters, kernel_size, padding=padding)
        self.ReLU = nn.ReLU()
        self.BN = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        """ Get output of block.
        x: tensor of shape (batch_size, input_channels, sequence_length), containing input data."""

        x = self.conv(x)
        x = self.ReLU(x)
        x = self.BN(x)

        return x


def construct_action_space(max_index, ss='BlockQNN'):
    action_space = []

    index = max_index

    if ss == 'BlockQNN':
        layers = list(set(TYPES.keys()))  # Allowed layer types for actions
        layers.remove(0)
        kernel_sizes = list(
            set().union(*[[0], KERNEL_SIZES_CONV, KERNEL_SIZES_POOL]))  # Allowed kernel sizes for actions
        preds = list(range(max_index))  # Allowed predecessors for actions

        for layer in layers:
            for kernel_size in kernel_sizes:
                for pred1 in preds:
                    for pred2 in preds:
                        code = NSC((index, layer, kernel_size, pred1, pred2))
                        if code.valid():
                            action_space.append((layer, kernel_size, pred1, pred2))

    if ss == 'Streamlined':
        layers = list(set(TYPES.keys()))
        layers.remove(0)
        kernel_sizes = list(set().union(*[[0], KERNEL_SIZES_CONV, KERNEL_SIZES_POOL]))
        kernel_numbers = list(set().union(*[[0], N_KERNELS_CONV]))
        preds = list(range(max_index))

        print(layers, kernel_sizes, kernel_numbers, preds)

        for layer in layers:
            for kernel_size in kernel_sizes:
                for n_kernels in kernel_numbers:
                    for pred1 in preds:
                        for pred2 in preds:
                            code = NSC((index, layer, kernel_size, pred1, pred2, n_kernels))
                            if code.valid():
                                action_space.append((layer, kernel_size, pred1, pred2, n_kernels))

    if ss == 'FeedForward':
        layers = list(set(TYPES.keys()))
        layers.remove(0)
        layers.remove(3)
        kernel_sizes = list(set().union(*[[0], KERNEL_SIZES_CONV, KERNEL_SIZES_POOL]))
        kernel_numbers = list(set().union(*[[0], N_KERNELS_CONV]))

        for index in range(1, max_index+1):
            for layer in layers:
                for kernel_size in kernel_sizes:
                    for n_kernels in kernel_numbers:
                        code = NSC((index, layer, kernel_size, index - 1, 0, n_kernels))
                        if code.valid():
                            action_space.append((layer, kernel_size, index - 1, 0, n_kernels))

    return action_space, len(action_space)


def find_search_space_size(maxindex, action_space, feedforward=False):
    for action in action_space:
        action = list(action)
        action[1] = list(TYPES.keys())[action[0]]

    n_actions = []

    for i in range(1, maxindex + 1):
        counter = 0
        for action in action_space:
            index = [i]
            index.extend(action)
            code = NSC(index)
            print(code.tuple, code.valid())
            if code.valid():
                if feedforward == False:
                    counter += 1
                elif feedforward == True and code.pred1 == index[0] - 1:
                    counter += 1
                elif feedforward == True and code.type == 'Terminal':
                    counter += 1
        n_actions.append(counter)
    print(n_actions)
    ss_size = np.prod(n_actions, dtype=np.double)

    return ss_size


if __name__ == '__main__':
    act_space, nA = construct_action_space(8, ss='FeedForward')

    print('Total allowed actions:', nA)
    print('Allowed actions:', act_space)

    search_space_size = find_search_space_size(8, act_space, feedforward=True)

    print('Search space size:', search_space_size)
