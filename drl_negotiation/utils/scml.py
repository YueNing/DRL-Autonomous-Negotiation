from drl_negotiation.utils.utils import parse_args, logging_setup
from drl_negotiation.core.hyperparameters import RUNNING_AGENT_TYPES
from scml.scml2020 import SCML2020World, DecentralizingAgent, IndDecentralizingAgent
import pandas as pd
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from negmas.helpers import get_class
from drl_negotiation.utils.utils import get_world_config


def _run(arglist, agent_types, world_config=None):
    if world_config is None:
        world = SCML2020World(
            **SCML2020World.generate(
                agent_types=agent_types,
                n_steps=100
            ),
            construct_graphs=True
        )
    else:
        world_config["agent_params"] = world_config["agent_params"][:-2]
        new_configuration = SCML2020World.generate(
            **world_config
        )
        world = SCML2020World(
            **new_configuration
        )
    world.run_with_progress()
    winner = world.winners[0]
    return world, winner


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


def run_scml(world_config=None):
    arglist = parse_args()
    logging_setup(logging.getLevelName(arglist.logging_level))
    if world_config is None:
        agent_types = [get_class(_) for _ in RUNNING_AGENT_TYPES]
    else:
        world_config = get_world_config(world_config)
        agent_types = world_config['agent_types']
    world, winner = _run(arglist, agent_types, world_config)
    logging.info(f"winner is {winner}")
    show(world, winner)
    show_scores(world)