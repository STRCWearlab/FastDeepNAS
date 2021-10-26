import torch 

# DQN config
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64       # minibatch size
GAMMA = 1           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4       # how often to update the network
DEVICE = torch.device("cuda")  # Device to run the network on
