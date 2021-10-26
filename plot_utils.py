import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def plot_scores(scores, rolling_window=100):
    """Plot scores and optional rolling mean using specified window."""
    plt.plot(scores)
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    cumulative_max = pd.Series(scores).cummax()
    plt.plot(cumulative_max)
    plt.savefig('rewards_over_time')
    return rolling_mean, cumulative_max


def plot_final_performance_vs_params(final_perf, search_params):
    """ final_perf must be an array of tuples containing (final_acc, final_f1, inf_time)
    search_params must be an array of the same length containing some parameters (either search space or controller) """

    plt.figure(2)
    plt.title('Final performance vs search space parameters.')
    plt.xlabel('search_params')
    plt.ylabel('final_perf')
    plt.plot(search_params, final_perf)
    plt.savefig('final_perf')


def plot_evaluated_models(env):
    evaluated_model_params = env.evaluated_models.keys()
    evaluated_model_stats = env.evaluated_models.values()

    keys = []
    values = []
    for key in env.evaluated_models.keys():
        keys.append(key)
        values.append(env.evaluated_models[key])

    rewards = [stats['reward'] for stats in values]
    repeats = [stats['n_repeats'] for stats in values]

    two_layer_keys = [True if key[0] == 2 else False for key in keys]
    one_layer_keys = [True if key[0] == 1 else False for key in keys]
    zero_layer_keys = [True if key[0] == 1 else False for key in keys]

    two_layer_units = np.array([key[1] for key in np.array(keys)[two_layer_keys]]).T
    one_layer_units = np.array([key[1] for key in np.array(keys)[one_layer_keys]]).T
    zero_layer_units = np.array([key[1] for key in np.array(keys)[zero_layer_keys]]).T

    two_layer_rewards = np.array(rewards)[two_layer_keys]
    two_layer_repeats = np.array(repeats)[two_layer_keys]
    one_layer_rewards = np.array(rewards)[one_layer_keys]
    one_layer_repeats = np.array(repeats)[one_layer_keys]
    zero_layer_rewards = np.array(rewards)[zero_layer_keys]
    zero_layer_repeats = np.array(repeats)[zero_layer_keys]

    reward_matrix = np.zeros((34, 34))
    repeats_matrix = np.zeros((34, 34), dtype=int)

    for i, reward in enumerate(two_layer_rewards):
        reward_matrix[two_layer_units[0][i]][two_layer_units[1][i]] = reward
    for i, repeated in enumerate(two_layer_repeats):
        repeats_matrix[two_layer_units[0][i]][two_layer_units[1][i]] = repeated

    for i, reward in enumerate(one_layer_rewards):
        reward_matrix[one_layer_units[0][i]][0] = reward
    for i, repeated in enumerate(one_layer_repeats):
        repeats_matrix[one_layer_units[0][i]][0] = repeated

    for i, reward in enumerate(zero_layer_rewards):
        reward_matrix[0][0] = reward
    for i, repeated in enumerate(zero_layer_repeats):
        repeats_matrix[0][0] = repeated

    plt.figure(1)
    reward_heatmap = sn.heatmap(reward_matrix, vmin=-0.1)
    plt.savefig('rewards_heatmap')
    plt.figure(2)
    repeat_heatmap = sn.heatmap(repeats_matrix, vmax=100)
    plt.savefig('repeats_heatmap')
