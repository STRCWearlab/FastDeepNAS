if __package__ is None or __package__ == '':
    # uses current directory visibility
    from nas_utils import *
else:
    # uses current package visibility
    from .nas_utils import *

import sklearn.metrics as metrics
from datetime import datetime


def test(model, X_test, y_test, batch_size=1000, verbosity='silent', log_name='log'):
    test_stats = np.unique(y_test, return_counts=True)[1]

    if verbosity == 'Full':
        print('Testing set statistics:')
        print(len(test_stats), 'classes with distribution', test_stats)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    test_losses = []
    targets_cumulative = []
    top_classes = []

    start = datetime.now()

    with torch.no_grad():
        with open('{}.csv'.format(log_name), 'a', newline='') as csvfile:
            for batch in iterate_minibatches(X_test, y_test, batch_size, num_batches=None):
                x, y = batch

                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                targets_cumulative.extend(y)

                inputs, targets = inputs.cuda(), targets.cuda()

                output = model(inputs)

                test_loss = criterion(output, targets.long())
                test_losses.append(test_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                top_classes.extend([top_class.item() for top_class in top_class.cpu()])

            equals = [top_classes[i] == target for i, target in enumerate(targets_cumulative)]
            accuracy = np.mean(equals)

            f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
            f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')

            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([np.mean(test_losses), accuracy, f1score, f1macro])

    end = datetime.now()

    time = (end - start).total_seconds()
    return (accuracy, f1score, f1macro), time, targets_cumulative, top_classes


def test_causal(model, X_test, y_test, stride, batch_size=1000, verbosity='silent', log_name='log'):

    test_stats = np.unique([a for y in y_test for a in y], return_counts=True)[1]

    if verbosity == 'Full':
        print('Testing set statistics:')
        print(len(test_stats), 'classes with distribution', test_stats)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    test_losses = []
    targets_cumulative = []
    top_classes = []

    start = datetime.now()

    with torch.no_grad():
        with open('{}.csv'.format(log_name), 'a', newline='') as csvfile:
            for batch in iterate_minibatches_2D(X_test, y_test, batch_size, stride, batchlen=5, drop_last=True):
                x, y, pos = batch

                inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                targets_cumulative.extend([y for y in y])

                inputs, targets = inputs.cuda(), targets.cuda()

                if pos == 0:
                    test_h = model.init_hidden(batch_size)

                output, test_h = model(inputs, test_h)

                test_loss = criterion(output, targets.long())
                test_losses.append(test_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                top_classes.extend([top_class.item() for top_class in top_class.cpu()])

            equals = [top_classes[i] == target for i, target in enumerate(targets_cumulative)]
            accuracy = np.mean(equals)

            f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
            f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')

            writer = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([np.mean(test_losses), accuracy, f1score, f1macro])

    end = datetime.now()

    time = (end - start).total_seconds()
    return (accuracy, f1score, f1macro), time, targets_cumulative, top_classes
