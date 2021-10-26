import random
from collections import namedtuple, deque
import nn_predictor as pred

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim

import utils
from config import *

N_KERNELS_CONV = [32, 64, 128, 256]
KERNEL_SIZES_CONV = [1, 5, 9, 15, 22]
KERNEL_SIZES_POOL = [2, 3, 5]


# This is our deep network architecture which will be used as the main and
# target network for the DQ agent
class DQN(nn.Module):

    def __init__(self, n_inputs, n_outputs, seed, hidden_size=64, n_layers=2):
        super(DQN, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.seed = torch.manual_seed(seed)

        self.rnn = nn.RNN(n_inputs, hidden_size, n_layers)

        self.predictor = nn.Linear(hidden_size, n_outputs)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = x.view(1, -1, self.n_inputs)

        x, _ = self.rnn(x)

        x = self.predictor(x)

        x = x.view(1, -1, self.n_outputs)

        x = self.output(x)

        return x


class DiscreteRNNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, action_space, seed, encoding='int', reward_shaping=False,
                 hidden_size=64, n_layers=2, priority=False, n_kernels_conv=[32, 64, 128, 256],
                 kernel_sizes_conv=[1, 5, 9, 15, 22], kernel_sizes_pool=[2, 3, 5], use_predictor=False,
                 pred_input_dim=None, load_path='trained_predictor_mlp.pt', epsilon_schedule='step'):
        """Initialize an Agent object.

        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.encoding = encoding

        # Q-Network
        self.qnetwork_local = DQN(state_size, action_size, seed, hidden_size, n_layers).to(DEVICE)
        self.qnetwork_target = DQN(state_size, action_size, seed, hidden_size, n_layers).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, priority)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Setup interim memory for reward shaping if required
        self.reward_shaping = reward_shaping
        if self.reward_shaping:
            self.interim_memory = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

        self.episode_counter = 0

        if epsilon_schedule == 'step':
            self.epsilon_schedule = np.concatenate((np.ones(5200), np.full(400, 0.9), np.full(400, 0.8),
                                                np.full(400, 0.7), np.full(400, 0.6), np.full(600, 0.5),
                                                np.full(600, 0.4), np.full(600, 0.3), np.full(600, 0.2),
                                                np.full(810, 0.1)))
        if epsilon_schedule == '20ksked':
            self.epsilon_schedule = np.concatenate((np.ones(5000), np.linspace(1, 0.1, 5000),
                                                    np.linspace(0.1, 0.05, 5000), np.linspace(0.05, 0, 5000),
                                                    np.zeros(1000)))
        else:
            self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule[self.episode_counter]

        self.priority = priority
        self.loss = 0

        self.action_space = action_space
        self.n_kernels_conv = n_kernels_conv
        self.kernel_sizes_conv = kernel_sizes_conv
        self.kernel_sizes_pool = kernel_sizes_pool

        if use_predictor:
            self.use_predictor = True

        if use_predictor == 'CNN':
            state_dict = torch.load(load_path)
            self.predictor = pred.BranchedPredictor(pred_input_dim, n_units=64).cuda().double()
            self.predictor.load_state_dict(state_dict)
            self.use_struct = False
        if use_predictor == 'MLP':
            state_dict = torch.load(load_path)
            self.predictor = pred.Predictor(pred_input_dim, n_units=64).cuda().double()
            self.predictor.load_state_dict(state_dict)
            self.use_struct = False

    def step(self, state, action, reward, next_state, done):

        if self.use_predictor and reward not in (0, -1):
            reward = self.predict(reward)

        if self.reward_shaping:
            # Save experiences in interim memory to implement reward shaping
            self.add_to_interim_memory(state, action, reward, next_state, done)
            if done:
                # Do reward shaping
                self.shape_rewards(reward)
                # Save experience in replay memory with shaped reward
                self.add_to_experience_memory(self.interim_memory)
        else:
            self.memory.add(state, action, reward, next_state, done)

        if done:
            self.epsilon = self.epsilon_schedule[self.episode_counter]
            self.episode_counter += 1

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

        return reward

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state += 1
        state = torch.from_numpy(np.array(state)).float().to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection for binary-encoded states
        if random.random() > self.epsilon:
            return np.argmax(action_values.squeeze(1).squeeze(0).cpu().data.numpy())
        else:
            return self.random_sample(state)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get state action values from local net
        expected_state_action_values = self.qnetwork_local(states).squeeze(0).gather(1, actions)

        # Compute next state action values using target net
        next_state_values = self.qnetwork_target(next_states).squeeze(0).max(1)[0].detach()

        target_state_action_values = ((1 - dones.squeeze(1)) * next_state_values * gamma) + rewards.squeeze(1)

        # Calculate TD delta for prioritized experience replay
        if self.priority:
            delta = (target_state_action_values - expected_state_action_values.squeeze(1)).detach()
            self.memory.update_deltas(delta.cpu())  # Update memory buffer with new deltas

        # Implement Huber loss function to approximate the Bellman equation
        self.loss = F.smooth_l1_loss(expected_state_action_values.squeeze(1), target_state_action_values)

        self.optimizer.zero_grad()  # Prepare gradients
        self.loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def predict(self, reward, type='MLP'):
        self.predictor.eval()
        if reward[1] is not None:
            with torch.no_grad():
                if type == 'MLP':
                    reward_flat = np.concatenate([np.array(reward[0]).T.flatten(), [np.log(reward[1])], [np.log(reward[2])]])
                    reward = torch.from_numpy(reward_flat).cuda()
                    reward = self.predictor(reward)
            return reward.detach().cpu()
        else:
            return 0

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(tau * local_param.data +
                                    (1.0 - tau) * target_param.data)

    def add_to_interim_memory(self, state, action, reward, next_state, done):
        self.interim_memory['states'].append(state)
        self.interim_memory['actions'].append(action)
        self.interim_memory['rewards'].append(reward)
        self.interim_memory['next_states'].append(next_state)
        self.interim_memory['dones'].append(done)

    def shape_rewards(self, reward):
        # Shape reward as rt = rT / T
        T = len(self.interim_memory['states'])
        N = len([m for m in self.interim_memory['rewards'] if m >= 0])
        for t in range(T):
            if self.interim_memory['rewards'][t] >= 0:
                self.interim_memory['rewards'][t] = reward / N

    def add_to_experience_memory(self, exp):
        T = len(exp['states'])
        for t in range(T):
            self.memory.add(exp['states'][t], exp['actions'][t], exp['rewards'][t],
                            exp['next_states'][t], exp['dones'][t])
            self.interim_memory = {'states': [], 'actions': [], 'rewards': [],
                                   'next_states': [], 'dones': []}

    def save_state(self, episode, path='dqn_checkpoint.pt'):
        torch.save({'epoch': episode,
                    'local_state_dict': self.qnetwork_local.state_dict(),
                    'target_state_dict': self.qnetwork_target.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.loss}, 'model_params/{}'.format(path))

    def random_sample(self, index):

        while True:
            kernel_size = 0
            n_kernels = 0
            Pred1 = 0
            Pred2 = 0

            layer_type = np.random.choice((1, 2, 3, 4))

            if layer_type == 1:
                n_kernels = np.random.choice(self.n_kernels_conv)

            if layer_type == 1:
                kernel_size = np.random.choice(self.kernel_sizes_conv)

            if layer_type == 2:
                kernel_size = np.random.choice(self.kernel_sizes_pool)

            if index > 1:
                if layer_type in (1, 2, 3):
                    Pred1 = np.random.choice(list(range(int(index))))
                if layer_type == 3:
                    Pred2 = np.random.choice(list(range(int(index))))

            layer = (layer_type, kernel_size, Pred1, Pred2, n_kernels)

            if len(np.argwhere([action == layer for action in self.action_space])) > 0:
                break

        return np.argwhere([action == layer for action in self.action_space])[0][0]


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, priority=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "delta"])
        random.seed(seed)
        self.generator = np.random.default_rng(seed)
        self.eps = 0.01  # Small probability offset to help prevent oscillation between large TD error samples

        self.priority = priority
        if self.priority:
            self.sampled_experiences = None

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, 1)
        self.memory.append(e)

    def sample(self, priority=False):
        """Randomly sample a batch of experiences from memory."""
        if priority:
            weights = \
            utils.normalized([np.abs(e.delta) + self.eps for e in self.memory if e is not None], order=1, axis=0)[
                0]
            choices = self.generator.choice(len(self.memory), replace=False, p=weights,
                                            size=self.batch_size)

        else:
            choices = self.generator.choice(len(self.memory), replace=False, size=self.batch_size)

        sampled_experiences = [self.memory[choice] for choice in choices]

        if priority:
            for exp in sampled_experiences:
                self.memory.remove(exp)  # Remove the experiences to be replaced with updated deltas
            self.sampled_experiences = sampled_experiences

        states = torch.from_numpy(np.vstack([e.state for e in sampled_experiences if e is not None])).float().to(
            DEVICE)
        actions = torch.from_numpy(np.vstack([e.action for e in sampled_experiences if e is not None])).long().to(
            DEVICE)
        rewards = torch.from_numpy(np.vstack([e.reward for e in sampled_experiences if e is not None])).float().to(
            DEVICE)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in sampled_experiences if e is not None])).float().to(DEVICE)
        dones = torch.from_numpy(
            np.vstack([e.done for e in sampled_experiences if e is not None]).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def update_deltas(self, delta):
        states = [e.state for e in self.sampled_experiences if e is not None]
        actions = [e.action for e in self.sampled_experiences if e is not None]
        rewards = [e.reward for e in self.sampled_experiences if e is not None]
        next_states = [e.next_state for e in self.sampled_experiences if e is not None]
        dones = [e.done for e in self.sampled_experiences if e is not None]

        for i, delta in enumerate(delta):
            self.memory.append(self.experience(states[i], actions[i], rewards[i],
                                               next_states[i], dones[i], delta))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
