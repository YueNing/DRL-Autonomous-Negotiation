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
        # between 0 and 1
        current_time = agent.awi.current_step / agent.awi.n_steps

        number_buyers, number_sellers = agent.running_negotiations_count
        number_request_buys, number_request_sells = agent.negotiation_requests_count

        my_input_product = agent.awi.my_output_product
        my_output_product = agent.awi.my_output_product
        number_buy_contracts = 0
        number_sell_contracts = 0

        if agent.contracts:
            for c in agent.contracts:
                if c.annotation['buyer'] == agent.id:
                    number_buy_contracts += 1
                if c.annotation['seller'] == agent.id:
                    number_sell_contracts += 1

        _obs.append(current_time)
        if seller:
            price_product = agent.awi.catalog_prices[my_output_product]
            number_contracts = number_sell_contracts
            _obs.append(number_buyers)
            _obs.append(number_request_buys)
            _obs.append(price_product)
            _obs.append(number_contracts)
        else:
            price_product = agent.awi.catalog_prices[my_input_product]
            number_contracts = number_buy_contracts
            _obs.append(number_sellers)
            _obs.append(number_request_sells)
            _obs.append(price_product)
            _obs.append(number_contracts)

        return np.array(_obs)

    def reward(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        # sub-goal, best deal which is defined as being nearest to the agent needs with lowest price
        # main-goal, maximum profitability at the end of episode.
        rew = 0
        rew += world.scores()[agent.id]
        if RANDOM_REWARD:
            return random.random()
        else:
            return rew

    def done(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return False

    def benchmark_data(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return {}