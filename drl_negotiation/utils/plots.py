import numpy as np
import seaborn as sns
import chart_studio.plotly as py
import plotly.io as pio
#pio.renderers.default = "browser"
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scml.scml2020 import SCML2020World
from drl_negotiation.core.hyperparameters import PLOT_BACKEND


def show_ep_rewards(data, model, number_episodes=20, extra=False, backend=PLOT_BACKEND):
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
    assert len(data) > 0, "data is empty"
    if model is None:
        save_times = len(data)
        save_rate = int(number_episodes / save_times)
    else:
        save_rate = model.save_rate
        number_episodes = model.num_episodes

    x_axis = np.arange(save_rate, number_episodes + save_rate, save_rate)
    y_axis = np.array(data)
    names = ["mean_episode_reward"]
    if extra:
        y_axis = np.reshape(y_axis, (2, int(len(data) / 2)))
        names +=["mean_episode_extra_reward"]
    assert len(x_axis) == y_axis.shape[1]

    data = pd.DataFrame(
        {
            **{'episode': x_axis,},
            **{names[e]:y_axis[e] for e in range(y_axis.shape[0])},
        }
    )
    if backend == "sns":
        sns.lineplot(
            x="episode", y="mean_episode_reward", hue='type',
            data=pd.melt(data, ['episode'], var_name="type", value_name="mean_episode_reward")
        )
        plt.show()
    elif backend == "plotly":
        fig = px.line(data, x="episode", y=names)
        #fig.show()
        py.plot(fig, filename='show_ep_reward', auto_open=True)
        pio.write_html(fig, file="show_ep_reward.html", auto_open=True)
    else:
        raise NotImplementedError

def show_agent_rewards(data, model:"MADDPGModel"=None, agents=5, number_episodes=20, extra=False, backend=PLOT_BACKEND):
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
        agents_name = [a.name for a in model.env.agents]
        if extra:
            agents += len(model.env.heuristic_agents)
            agents_name += [a.name for a in model.env.heuristic_agents]
        save_times = int(len(data) / agents)

    assert save_rate * save_times == number_episodes
    trainable_agents_data = np.reshape(np.array(data[:len(model.env.agents)*save_times]),
                                       (save_times, len(model.env.agents)))
    extra_agents_data = np.reshape(np.array(data[len(model.env.agents)*save_times:]),
                                   (save_times, len(model.env.heuristic_agents)))
    x_axis = np.arange(save_rate, number_episodes + save_rate, save_rate)
    assert len(x_axis) == save_times
    #y_axis = np.reshape(np.array(data), (save_times, agents))
    data = pd.DataFrame(
        {
            **{'episode': x_axis,},
            **{agents_name[agent]:trainable_agents_data[:, agent] for agent in range(len(model.env.agents))},
            **{agents_name[len(model.env.agents)+agent]:extra_agents_data[:, agent]
               for agent in range(len(model.env.heuristic_agents))}
        }
    )
    if backend=="sns":
        sns.lineplot(
            x="episode", y="mean_reward", hue='agent',
            data=pd.melt(data, ['episode'], var_name="agent", value_name="mean_reward")
        )
        plt.show()
    elif backend=="plotly":
        fig = px.line(data, x="episode", y=agents_name)
        fig.show()
    else:
        raise NotImplementedError

def show_scores(world: SCML2020World):
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


def cumulative_reward(data, model, extra=False, backend=PLOT_BACKEND):
    """
    cumulative episode reward
    >>> cumulative_reward([0.3, 0.6, 0.3, -0.2, 0.8, 1.0])

    Args:
        data:

    Returns:

    """
    data = np.reshape(data, (2, int(len(data) / 2)))
    x_axis = np.arange(data.shape[1])
    y_axis = data[0]
    cum_sum_y_axis = np.cumsum(y_axis)
    y_axis_extra = data[1]
    if extra:
        cum_sum_y_extra_axis = np.cumsum(y_axis_extra)
        data = pd.DataFrame(
            {
                **{"episode": x_axis},
                **{
                    "ep_reward": y_axis,
                    "cumulative_reward": cum_sum_y_axis,},
                **{
                    "ep_reward_extra": y_axis_extra,
                    "cumulative_extra_reward": cum_sum_y_extra_axis,
                }
            }
        )
        y_names = ["ep_reward", "cumulative_reward", "ep_reward_extra", "cumulative_extra_reward"]
    else:
        data = pd.DataFrame(
            {
                **{"episode": x_axis},
                **{
                    "ep_reward": y_axis,
                    "cumulative_reward": cum_sum_y_axis,}
            }
        )
        y_names = ["ep_reward", "cumulative_reward"]
    if backend == "sns":
        sns.lineplot(x="episode", y="reward", hue="reward_type",
                     data=pd.melt(data, ['episode'], var_name="reward_type", value_name="reward"))
        plt.show()
    elif backend == "plotly":
        fig = px.line(data, x="episode", y=y_names)
        fig.show()
    else:
        raise NotImplementedError

if __name__ == '__main__':
    import doctest
    doctest.testmod()
