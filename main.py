import gym

from dqn import *
from plot_utils import *


def interact(agent, env, n_episodes=5000, max_index=8):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_by_net = {}

    for i_episode in range(1, n_episodes + 1):
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action, return_avg=False)
            reward = agent.step(state, action, reward, next_state, done)
            state = next_state
            if reward >= 0:
                score += float(reward)

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                            agent.epsilon), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}\tEpsilon: {:.2f}'.format(i_episode, np.mean(scores_window),
                                                                                agent.epsilon))
        if i_episode == 5000:
            agent.save_state(i_episode, path='exploration_done.pt')
            env.save_evaluated_models(name='evaluated_models_exploration_done.csv')

        scores_by_net[env.spec.hashable] = score

    return scores, scores_by_net


if __name__ == '__main__':
    n_epochs = 10

    env = gym.make('gym_nas_pt:nas_pt-v0', max_index=8, ch='all', sub='all', classifier='LSTM', use_redef_reward=False,
                   n_epochs_train=n_epochs, for_predictor=True)

    input_dim = n_epochs * 5 + 2
    Agent = DiscreteRNNAgent(1, env.action_size, env.nsc_space, seed=1,
                             n_kernels_conv=[32, 64, 128, 256], kernel_sizes_conv=[1, 2, 3, 5, 8],
                             kernel_sizes_pool=[2, 3, 5], use_predictor='MLP', pred_input_dim=input_dim)

    scores, scores_by_net = interact(Agent, env)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('scores')
