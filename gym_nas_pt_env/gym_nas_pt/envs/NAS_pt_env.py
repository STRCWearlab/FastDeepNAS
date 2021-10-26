import gym
from gym import spaces

if __package__ is None or __package__ == '':
    # uses current directory visibility
    from model_generator import *
    from NSC import *
    from config import *
    from train_model import *
    from nas_utils import *
    from graph_vis import *
    from preprocess_data import *
    from test_model import *
    from flops_bench_utils import *
else:
    # uses current package visibility
    from .model_generator import *
    from .NSC import *
    from .config import *
    from .train_model import *
    from .nas_utils import *
    from .graph_vis import *
    from .preprocess_data import *
    from .test_model import *
    from .flops_bench_utils import *

import torch
from ast import literal_eval as make_tuple
from thop import profile

try:
    import cPickle as pickle
except:
    import pickle

VERBOSITY = 'silent'  # set 'full' for debugging


class NASPtEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, max_index=11, state_encoding='int', action_encoding='int', conv_filters=N_KERNELS_CONV,
                 ch='fast', get_data=None, sub='fast', use_redef_reward=True, classifier='Linear', window_step=8,
                 window_size=16, n_epochs_train=15, for_predictor=False):

        channels_dict = {'fast': 12, 'all': 113}
        self.n_channels = channels_dict[ch]

        if get_data == 'Opportunity':
            dataset = find_data('Opportunity')
            generate_data('Opportunity', dataset, sub, 'gestures', ch)

        # Observations: the current layer index
        self.observation_space = spaces.Discrete(max_index)

        # Gen action space
        self.nsc_space, self.nA = construct_action_space(max_index, ss='Streamlined')
        print('Size of action space:', self.nA)
        self.action_space = spaces.Discrete(self.nA)
        self.max_index = max_index
        self.conv_filters = conv_filters
        self.ch = ch

        self.action_size = self.action_space.n
        self.state_size = self.observation_space.n

        self.state_encoding = state_encoding  # Determines how the state should be presented
        self.action_encoding = action_encoding  # , and how the env expects the action to be represented

        # Hyperparameters and parameters for training
        self.args = {'window_size': window_size, 'batch_size': 1000, 'num_batches': 20,
                     'num_batches_val': 10, 'log_name': 'log' + '.csv', 'patience': 20,
                     'window_step': window_step}

        if classifier == 'LSTM':
            self.X_train, self.y_train = load_data('{}_train'.format(ch), self.args['window_size'],
                                                   self.args['window_step'], keep_seperate=True)
            self.X_val, self.y_val = load_data('{}_val'.format(ch), self.args['window_size'], self.args['window_step'],
                                               keep_seperate=True)
        else:
            self.X_train, self.y_train = load_data('{}_train'.format(ch), self.args['window_size'],
                                                   self.args['window_step'])
            self.X_val, self.y_val = load_data('{}_val'.format(ch), self.args['window_size'], self.args['window_step'])

        self.evaluated_models = {}

        self.internal_state = []  # Initialise internal stored network state
        self.internal_state.append(NSC((0, 0, 0, 0, 0, 0)))

        self.current_index = self.internal_state[-1].index  # Initialise exposed representation of state

        self.use_redef_reward = use_redef_reward
        if use_redef_reward:
            if classifier == 'Linear':
                self.mu = 1
                self.rho = 100
            if classifier == 'LSTM':
                self.mu = 0
                self.rho = 0

        try:
            self.evaluated_models = self.load_evaluated_models()
        except FileNotFoundError:
            print('No previous results found, starting new memo for evaluated models.')
            pass

        self.classifier = classifier
        self.n_epochs_train = n_epochs_train
        self.for_predictor = for_predictor

    def evaluate(self, arch, return_avg=5, return_struct=False, verbosity=VERBOSITY):

        """ Call an external script to evaluate a pytorch model on the target dataset.
            Return done = True if criteria has not improved in tolerance iterations. """

        n_epochs = self.n_epochs_train

        # Append a terminal layer if we don't have one
        self.spec = self.get_spec()

        try:
            self.model = self.spec.write(self.n_channels, self.args['window_size'], self.args['batch_size'],
                                         self.classifier).cuda()
        # We will get a zero division error if there are too many pooling layers
        except ZeroDivisionError:
            criteria = 0
            self.evaluated_models[self.spec.hashable] = (criteria, 0, None, None, None)

            if verbosity == 'full' or verbosity == 'quiet':
                print('Rejecting invalid model due to ZeroDivisionError, setting reward to {}.\
                       This can sometimes happen due to too many stacked \
                       pooling layers (~1/1000), but if it happens often \
                       there could be something else wrong.'.format(criteria))
            return criteria

        if self.spec.hashable not in self.evaluated_models.keys():
            try:
                if verbosity == 'full':
                    print('Evaluating architecture:', self.spec.hashable)

                if self.classifier == 'LSTM':
                    criteria, stdev, train_time = train_causal(self.model, self.X_train, self.y_train,
                                                               self.X_val, self.y_val, self.args['window_step'],
                                                               epochs=n_epochs, verbosity=verbosity,
                                                               return_avg=return_avg, return_struct=return_struct)

                else:
                    criteria, stdev, train_time = train(self.model, self.X_train, self.y_train,
                                                        self.X_val, self.y_val, epochs=n_epochs,
                                                        verbosity=verbosity, return_avg=return_avg,
                                                        return_struct=return_struct)
                density = self.get_density()
                FLOPs = self.get_flops()
                self.evaluated_models[self.spec.hashable] = (criteria, stdev, train_time, density, FLOPs)
                if self.use_redef_reward:
                    criteria = self.redefined_reward(criteria, stdev)
                if self.for_predictor:
                    criteria = (np.array(criteria), density, FLOPs)

            except RuntimeError:
                criteria = 0
                self.evaluated_models[self.spec.hashable] = (criteria, 0, None, None, None)
                if verbosity == 'full' or verbosity == 'quiet':
                    print('Rejecting invalid model due to RuntimeError, setting reward to {}.\
                           This can sometimes happen due to too many stacked \
                           pooling layers (~1/1000), but if it happens often \
                           there could be something else wrong. If it happens every time its \
                           probably a device mismatch.'.format(criteria))

        else:
            criteria, stdev, inf_time, density, FLOPs = self.evaluated_models[self.spec.hashable]
            if self.use_redef_reward:
                criteria = self.redefined_reward(criteria, stdev)
            if self.for_predictor:
                criteria = (np.array(criteria), density, FLOPs)

        return criteria

    def step(self, action, verbosity=VERBOSITY, n_avg=1, eval=True, return_avg=5, return_struct=False):

        """ Perform an action - in this case generate and evaluate a new architecture. """

        done = False
        info = ''

        action = self.action_to_nsc(action)

        # Add layer to the stored state
        if action.valid():
            self.internal_state.append(action)
            self.current_index = self.internal_state[-1].index

            if action.type == 'Terminal' or self.current_index == self.max_index:
                if eval:
                    reward = self.evaluate(self.internal_state, return_avg, return_struct)
                else:
                    reward = 0
                done = True
                return self.current_index, reward, done, info
            else:
                info = 'Valid layer, setting intermediate reward to 0'
                reward = 0  # Static reward associated with each valid layer
                return self.current_index, reward, done, info
        else:
            info = 'Invalid layer, setting intermediate reward to -1'
            reward = -1  # Static reward associated with each invalid layer
            return self.current_index, reward, done, info

    def reset(self):
        # import torch
        # import gc
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass

        if 'model' in self.__dict__.keys():
            del self.model
        torch.cuda.empty_cache()

        self.internal_state = []  # Initialise internal stored network state
        self.internal_state.append(NSC((0, 0, 0, 0, 0)))

        self.current_index = self.internal_state[-1].index
        return self.current_index

    def test(self, model=None, verbosity='Full', logfile='log'):

        if model is None:
            self.spec = self.get_spec()
            density = self.get_density()
            model = self.spec.write(self.n_channels, self.args['window_size'], self.args['batch_size'],
                                    self.classifier).cuda()
        else:
            density = self.get_density(model)
            model = model.write(self.n_channels, self.args['window_size'], self.args['batch_size'],
                                self.classifier).cuda()

        if verbosity == 'Full':
            print('Testing', model)

        if self.classifier == 'LSTM':
            train_causal(model, self.X_train, self.y_train, self.X_val, self.y_val, self.args['window_step'],
                         epochs=300, verbosity=verbosity, early_stop=50, lr_schedule='step', logfile=logfile)
            X_test, y_test = load_data('{}_test'.format(self.ch), self.args['window_size'], self.args['window_step'],
                                       keep_seperate=True)
        else:
            _, _, _ = train(model, self.X_train, self.y_train, self.X_val, self.y_val, epochs=300, verbosity=verbosity,
                            early_stop=50, lr_schedule='step', logfile=logfile)
            X_test, y_test = load_data('{}_test'.format(self.ch), self.args['window_size'], self.args['window_step'])

        if self.classifier == 'LSTM':
            criteria_full, inf_time_full, targets_full, pred_full = test_causal(model, X_test, y_test, self.args['window_step'],
                                                            verbosity='Full')

            state_dict = torch.load('checkpoint.pt')
            model.load_state_dict(state_dict)

            criteria_early, inf_time_early, targets_early, pred_early = test_causal(model, X_test, y_test, self.args['window_step'],
                                                            verbosity='Full')
        else:
            criteria_full, inf_time_full, targets_full, pred_full = test(model, X_test, y_test, verbosity='Full')

            state_dict = torch.load('checkpoint.pt')
            model.load_state_dict(state_dict)

            criteria_early, inf_time_early, targets_early, pred_early = test(model, X_test, y_test, verbosity='Full')

        flops = self.get_flops(model)

        if criteria_early[1] >= criteria_full[1]:
            criteria, inf_time, targets, pred = criteria_early, inf_time_early, targets_early, pred_early
        else:
            criteria, inf_time, targets, pred = criteria_full, inf_time_full, targets_full, pred_full

        return criteria, inf_time, density, flops, targets, pred

    def render(self, model=None, mode='human', fname='output', ftype='png'):
        if model is None:
            model = self.spec

        graph = DNNGraph(model)
        print('Saving rendered graph to disk.')
        if ftype == 'svg':
            graph.graph.write_svg(fname + '.svg')
        if ftype == 'png':
            graph.graph.write_png(fname + '.png')

    def close(self):

        for key, obj in self.__dict__:
            print('Cleaning up {}'.format(key))
            del obj

        print('Clean up and close the environment. ')

    def action_to_nsc(self, action):

        index = self.current_index + 1
        layer, kernel_size, pred1, pred2, n_kernels = self.nsc_space[action]

        return NSC((index, layer, kernel_size, pred1, pred2, n_kernels))

    def get_spec(self, nsc_list=None):
        """ nsc list is a list of nodes in vector form i.e. (0,0,0,0,0)"""

        if nsc_list:
            nsc_list = [NSC(nsc) for nsc in nsc_list]
        else:
            nsc_list = self.internal_state

        if nsc_list[-1].type != 'Terminal':
            nsc_list.append(NSC((self.current_index + 1, 4, 0, 0, 0, 0)))

        spec = ModelSpec(nsc_list)

        return spec

    def get_flops(self, model=None):

        if model is None:
            model = self.model

        model = model.cuda().train()

        # Pass one batch through the network to record FLOPs
        if self.classifier == 'LSTM':
            for batch, _, _ in iterate_minibatches_2D(self.X_train, self.y_train, BATCH_SIZE, self.args['window_step'],
                                                   num_batches=1, batchlen=1):

                batch = torch.from_numpy(batch).cuda()
                h = model.init_hidden(BATCH_SIZE)
                macs, params = profile(model, inputs=(batch, h,), verbose=False)
        else:
            for batch, _ in iterate_minibatches(self.X_train, self.y_train, num_batches=1, batchsize=BATCH_SIZE):
                batch = torch.from_numpy(batch).cuda()
                macs, params = profile(model, inputs=(batch,), verbose=False)

        return macs / BATCH_SIZE  # Get results in FLOPs per sequence in the batch

    def get_density(self, model=None):

        if model is None:
            model = self.spec

        return model.density()

    def redefined_reward(self, reward, stdev):

        FLOPs = self.get_flops()
        density = self.get_density()

        if FLOPs != 0:
            return (reward * 100 + self.mu * np.log(FLOPs) + self.rho * stdev) / 100
        else:
            return (reward * 100 + self.rho * stdev) / 100

    def save_evaluated_models(self, name='evaluated_models.csv'):

        with open(name, 'wb') as file:
            pickle.dump(self.evaluated_models, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_evaluated_models(self, name='evaluated_models.csv'):

        with open(name, 'rb') as file:
            evaluated_models_dict = pickle.load(file)

        return evaluated_models_dict
