import logging
from drl_negotiation.utils.utils import parse_args, logging_setup
from drl_negotiation.core.hyperparameters import RUNNING_AGENT_TYPES
from scml.scml2020 import SCML2020World, DecentralizingAgent, IndDecentralizingAgent
from negmas.helpers import get_class
from drl_negotiation.utils.utils import get_world_config
from drl_negotiation.utils.plots import show, show_scores


def _run(arglist, agent_types, world_config=None):
    if world_config is None:
        world = SCML2020World(
            **SCML2020World.generate(
                agent_types=agent_types,
                n_steps=10
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


def run_scml(world_config=None):
    arglist = parse_args()
    logging_setup(logging.getLevelName(arglist.logging_level))
    world_config_path = world_config
    if world_config is None:
        agent_types = [get_class(_) for _ in RUNNING_AGENT_TYPES]
    else:
        world_config = get_world_config(world_config_path)
        agent_types = world_config['agent_types']
    world, winner = _run(arglist, agent_types, world_config)
    logging.info(f"winner is {winner}")
    show(world, winner)
    show_scores(world)