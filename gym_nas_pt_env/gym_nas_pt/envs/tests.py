from NAS_pt_env import *


def test_nsc_validity():
    test_codes = {(0, 0, 0, 0, 0): True,
                  (0, 1, 0, 0, 0): False,
                  (1, 1, 5, 0, 0): True,
                  (1, 1, 2, 0, 0): False,
                  (1, 1, 5, 1, 0): False,
                  (1, 1, 5, 0, 1): False,
                  (1, 2, 2, 0, 0): True,
                  (1, 2, 10, 0, 1): False,
                  (12, 1, 2, 3, 4): False,
                  (1, 5, 0, 0, 0): False,
                  (10, 5, 0, 3, 4): True,
                  (10, 5, 5, 0, 0): False}

    for code in test_codes.keys():
        print('Checking validity of code {}'.format(code))
        assert NSC(code).valid() == test_codes[code]
        print('Passed!')


def test_model_gen(nsc_list, data, labels):
    generator = ModelSpec(nsc_list)

    print(generator.encoding)

    model = generator.write(113, 1000, 'LSTM')
    model = model.cuda()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-8, amsgrad=True)

    model.train()

    criterion = nn.CrossEntropyLoss()

    for batch in iterate_minibatches(data, labels, BATCH_SIZE, num_batches=10):
        opt.zero_grad()

        x, y = batch

        inputs = torch.from_numpy(x).cuda()
        targets = torch.from_numpy(y).cuda()

        output = model(inputs)

        loss = criterion(output, targets.long())

        loss.backward()
        opt.step()

        print(loss.item())


def random_sample_NSC(ss='BlockQNN'):
    if ss == 'BlockQNN':
        net = [NSC((0, 0, 0, 0, 0))]

        index = 1

        while index < 11:

            layer_type = np.random.choice((1, 2, 3, 4, 5, 6, 7))

            if layer_type in (1, 2, 3):
                kernel_size = np.random.choice((1, 2, 3, 5, 7, 9, 11))
            else:
                kernel_size = 0

            if index > 1:
                Pred1 = np.random.choice(list(range(index - 1)))
                Pred2 = np.random.choice(list(range(index - 1)))

                layer = NSC((index, layer_type, kernel_size, Pred1, Pred2))

            else:
                layer = NSC((index, layer_type, kernel_size, 0, 0))

            if layer.valid():
                net.append(layer)
                index += 1

                if layer_type == 7:
                    break

        if net[-1].type != 'Terminal':
            net.append(NSC((net[-1].index + 1, 7, 0, 0, 0)))

    elif ss == 'Streamlined':
        net = [NSC((0, 0, 0, 0, 0, 0))]

        index = 1


        while index < 8:

            kernel_size = 0
            n_kernels = 0
            Pred1 = 0
            Pred2 = 0

            layer_type = np.random.choice((1, 2, 3, 4))


            if layer_type == 1:
                n_kernels = np.random.choice((32, 64, 128, 256))

            if layer_type in (1, 2):
                kernel_size = np.random.choice((1, 2, 5, 10, 30))

            if index > 1:
                if layer_type in (1, 2, 3):
                    Pred1 = np.random.choice(list(range(index - 1)))
                if layer_type == 3:
                    Pred2 = np.random.choice(list(range(index - 1)))

            layer = NSC((index, layer_type, kernel_size, Pred1, Pred2, n_kernels))
            if layer.valid():
                net.append(layer)
                index += 1

                if layer_type == 4:
                    break

        if net[-1].type != 'Terminal':
            net.append(NSC((net[-1].index + 1, 4, 0, 0, 0, 0)))
    return net


def test_get_flops_dens(env, nsc_list):
    generator = ModelSpec(nsc_list)
    print(generator.encoding)
    env.spec = generator
    env.model = generator.write(113, 15, 1000)

    return env.get_flops(), env.get_density()


def test_save_models(env):

    env.evaluated_models = {'((0,0,0,0,0),(1,3,5,0,0))': (0.05, 0.01, 0.5, 0.1),
                            '((0,0,0,0,0),(1,1,5,0,0))': (0.05, 0.11, 0.2, 0.8)}

    env.save_evaluated_models()
    env.reset()
    env.evaluated_models = env.load_evaluated_models()

    print(env.evaluated_models)


if __name__ == '__main__':
    env = gym.make('gym_nas_pt:nas_pt-v0', max_index=8, ch='all', sub='all', classifier='LSTM')
    nsc_list = random_sample_NSC(ss='Streamlined')
    test_model_gen(env, nsc_list)
