##############################################################
# scml sccenario, for concurrent negotiation control
##############################################################
from drl_negotiation.scenarios.scenario import BaseScenario
from drl_negotiation.core.hyperparameters import *
from drl_negotiation.utils.utils import make_world
from scml import SCML2020World
from drl_negotiation.core.core import MySCML2020Agent, TrainWorld
import numpy as np
import random

class Scenario(BaseScenario):
    def make_world(self, config):
        # configuration, for Scenario scml_concurrent
        world = make_world(config=config)
        return world

    def reset_world(self, world):
        agent_types = world.configuration['agent_types']
        agent_params = world.configuration['agent_params'][:-2]
        n_steps = world.configuration['n_steps']

        reset_configuration = SCML2020World.generate(
            agent_types=agent_types,
            agent_params=agent_params,
            n_steps=n_steps
        )

        reset_configuration['negotiation_speed'] = world.configuration['negotiation_speed'] \
            if "negotiation_speed" in world.configuration else NEGOTIATION_SPEED

        world.__init__(configuration=reset_configuration)

    def observation(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        _obs = []
        current_time = agent.awi.current_step / agent.awi.n_steps

        number_buyers, number_sellers = agent.running_negotiations_count

        if seller:
            price_product = None
            numbers_contracts = None
            _obs.append(number_buyers)
        else:
            price_product = None
            numbers_contracts = None
            _obs.append(number_sellers)

        return np.array(_obs)

    def reward(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        # sub-goal, best deal which is defined as being nearest to the agent needs with lowest price
        # main-goal, maximum profiability at the end of episode.
        return random.random()

    def done(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return False

    def benchmark_data(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return {}