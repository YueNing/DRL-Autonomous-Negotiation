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
        # between 0 and 1
        current_time = [agent.awi.current_step / agent.awi.n_steps]

        running = agent.running_negotiations_count
        requesting = agent.negotiation_requests_count

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

        last_offers = [[] for _ in agent.awi.my_consumers] + [[] for _ in agent.awi.my_suppliers]
        if seller:
            #last_offers = [[] for _ in agent.awi.my_consumers]
            for nid in agent.controllers[1].history_offers:
                negotiation = [negotiation for negotiation in agent.controllers[1].history_running_negotiations
                               if negotiation.negotiator == agent.controllers[1].negotiators[nid][0]]
                if negotiation:
                    if negotiation[0].annotation["seller"] == agent.id:
                        last_offers[sorted(agent.awi.my_consumers).index(negotiation[0].annotation["buyer"])].append(
                            agent.controllers[1].history_offers[nid]
                        )
                    else:
                        last_offers[len(agent.awi.my_consumers) + sorted(agent.awi.my_suppliers).index(
                            negotiation[0].annotation["seller"])].append(agent.controllers[1].history_offers[nid])

            agent.controllers[0].history_offers = {}

            for index, offer in enumerate(last_offers):
                if offer:
                    last_offers[index] = np.array(offer).mean(axis=0).tolist()
                else:
                    last_offers[index] = [0, 0, 0]

            price_product = [agent.awi.catalog_prices[my_output_product]]
            last_offers = np.array(last_offers).flatten().tolist()
            print(f"seller last_offers {last_offers}")
        else:
            last_offers = [[] for _ in agent.awi.my_suppliers]
            for nid in agent.controllers[0].history_offers:
                negotiation = [negotiation for negotiation in agent.controllers[0].history_running_negotiations if negotiation.negotiator ==
                               agent.controllers[0].negotiators[nid][0]]
                if negotiation:
                    if negotiation[0].annotation["buyer"] == agent.id:
                        last_offers[len(agent.awi.my_consumers) + sorted(agent.awi.my_suppliers).index(
                            negotiation[0].annotation["seller"])].append(
                            agent.controllers[0].history_offers[nid])
                    else:
                        last_offers[sorted(agent.awi.my_consumers).index(negotiation[0].annotation["buyer"])].append(
                            agent.controllers[0].history_offers[nid])

            agent.controllers[0].history_offers = {}

            for index, offer in enumerate(last_offers):
                if offer:
                    last_offers[index] = np.array(offer).mean(axis=0).tolist()
                else:
                    last_offers[index] = [0, 0, 0]

            price_product = [agent.awi.catalog_prices[my_input_product]]
            last_offers = np.array(last_offers).flatten().tolist()
            print(f"buyer last_offers {last_offers}")

        if seller:
            return np.concatenate((current_time, last_offers, running, requesting,
                                   [number_buy_contracts, number_sell_contracts], price_product))
        else:
            return np.concatenate((current_time, last_offers, running, requesting,
                                   [number_buy_contracts, number_sell_contracts], price_product))

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