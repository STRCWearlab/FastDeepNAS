import numpy as np


def iterate_minibatches(inputs, targets, batchsize, shuffle=True, num_batches=None):
    batch = lambda j: [x for x in range(j * batchsize, (j + 1) * batchsize)]

    batches = [i for i in range(int(len(inputs) / batchsize))]

    if shuffle:
        np.random.shuffle(batches)
        for i in batches[:num_batches]:
            yield np.array([inputs[i] for i in batch(i)]), np.array([targets[i] for i in batch(i)])

    else:
        for i in batches[:num_batches]:
            yield np.array([inputs[i] for i in batch(i)]), np.array([targets[i] for i in batch(i)])


def normalized(a, axis=-1, order=2):
    """ Get vector norm of a """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)
