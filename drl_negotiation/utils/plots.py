import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def show_ep_rewards(data, model, number_episodes=20):
    """
    mean episode reward
    >>> show_ep_rewards([42.36277500051694, 43.57323638074746, 44.766200950337335, 43.46595151832854],
    ...                 None,
    ...                 number_episodes=20
    ...             )

    Args:
        data:
        model:
        number_episodes:

    Returns:

    """
    if model is None:
        save_times = len(data)
        save_rate = int(number_episodes / save_times)
    else:
        save_rate = model.save_rate
        number_episodes = model.num_episodes

    x_axis = np.arange(save_rate, number_episodes + save_rate, save_rate)
    y_axis = np.array(data)
    assert len(x_axis) == len(y_axis)
    data = pd.DataFrame(
        {"episode":x_axis, "mean_episode_reward": y_axis}
    )
    sns.lineplot(x="episode", y="mean_episode_reward", data=data)
    plt.show()


def show_agent_rewards(data, model:"MADDPGModel"=None, agents=5, number_episodes=20):
    """
    >>> show_agent_rewards([8.409011336587847, 8.351518352218934, 9.003777340184879, 8.27788729251806, 8.32058067900721,
    ...                     8.27440019409948, 9.02914681190003, 8.430564051323255, 8.954079046561578, 8.885046276863111,
    ...                     9.275983476423741, 8.454406683479911, 9.050808573158575, 8.965107293250188, 9.019894924024928,
    ...                     8.68512853968182, 8.491117400241738, 8.731224550428184, 9.311305738783572, 8.247175289193219],
    ...                     None, agents=5, number_episodes=20)

    Args:
        data:
        model:
        agents:
        number_episodes:
    Returns:

    """
    if model is None:
        if type(agents) == int:
            save_times = int(len(data) / agents)
            save_rate = number_episodes / save_times
            agents_name = ['agent' + str(a) for a in range(agents)]
    else:
        save_rate = model.save_rate
        number_episodes = model.num_episodes
        agents = len(model.env.agents)
        save_times = int(len(data) / agents)
        agents_name = [a.name for a in model.env.agents]

    assert save_rate * save_times == number_episodes

    x_axis = np.arange(save_rate, number_episodes + save_rate, save_rate)
    assert len(x_axis) == save_times
    y_axis = np.reshape(np.array(data), (save_times, agents))
    data = pd.DataFrame(
        {
            **{'episode': x_axis,},
            **{agents_name[agent]:y_axis[:, agent] for agent in range(agents)}
        }
    )
    sns.lineplot(
        x="episode", y="mean_reward", hue='agent',
        data=pd.melt(data, ['episode'], var_name="agent", value_name="mean_reward")
    )
    plt.show()


def show_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()


def show(world, winner):
    stats = pd.DataFrame(data=world.stats)
    fig, axs = plt.subplots(2, 3)
    for ax, key in zip(axs.flatten().tolist(), ["score", "balance", "assets", "productivity",
                                                "spot_market_quantity", "spot_market_loss"]):
        ax.plot(stats[f"{key}_{winner}"])
        ax.set(ylabel=key)
    fig.show()


def cumulative_reward(data):
    """
    cumulative episode reward
    >>> cumulative_reward([0.3, 0.6, 0.3, -0.2, 0.8, 1.0])

    Args:
        data:

    Returns:

    """
    x_axis = np.arange(len(data))
    y_axis = data
    cum_sum_y_axis = np.cumsum(y_axis)

    data = pd.DataFrame(
        {
            **{"episode": x_axis},
            **{
                "ep_reward": y_axis,
                "cumulative_reward": cum_sum_y_axis
            }
        }
    )

    sns.lineplot(x="episode", y="reward", hue="reward_type",
                 data=pd.melt(data, ['episode'], var_name="reward_type", value_name="reward"))
    plt.show()


if __name__ == '__main__':
    import doctest
    doctest.testmod()