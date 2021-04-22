##############################################################
# scml sccenario, for concurrent negotiation control
##############################################################
from drl_negotiation.scenarios.scenario import BaseScenario
from drl_negotiation.core.hyperparameters import *
from drl_negotiation.utils.utils import make_world
from scml import SCML2020World
from drl_negotiation.core.core import TrainWorld
from drl_negotiation.core.core import MySCML2020Agent, TrainWorld
import numpy as np
import random

class Scenario(BaseScenario):
    def make_world(self, config):
        # configuration, for Scenario scml_concurrent
        world = make_world(config=config)
        return world

    def reset_world(self, world:TrainWorld):
        world = make_world(config=world.configuration)
        return world
        # world.__init__(configuration=world.configuration)

    def observation(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        last_offers = [[] for _ in agent.awi.my_consumers] + [[] for _ in agent.awi.my_suppliers]
        negotiation_ranges = [[] for _ in agent.awi.my_consumers] + [[] for _ in agent.awi.my_suppliers]

        # observe the range of negotiation
        for negotiation in agent.running_negotiations:
            logging.debug(f"{agent}:{negotiation.annotation}")
            partner = negotiation.annotation["buyer"] if negotiation.annotation["seller"] == agent.id else negotiation.annotation["seller"]
            negotiation_ranges[sorted(agent.awi.my_consumers + agent.awi.my_suppliers).index(partner)].append(negotiation.negotiator.ami.issues)

        # observe the last offer proposed by the negotiation partner
        if hasattr(agent, "controllers"):
            for is_seller, controller in agent.controllers.items():
                for nid in controller.history_offers:
                    negotiation = [negotiation for negotiation in controller.history_running_negotiations if negotiation.negotiator == controller.negotiators[nid][0]]
                    if negotiation:
                        if negotiation[0].annotation["seller"] == agent.id:
                            partner = negotiation[0].annotation["buyer"]
                        else:
                            partner = negotiation[0].annotation["seller"]

                        # offers proposed by partner
                        last_offers[sorted(agent.awi.my_consumers + agent.awi.my_suppliers).index(partner)].append(controller.history_offers[nid])
                controller.history_offers = {}

        price_product = [agent.awi.catalog_prices[agent.awi.my_output_product if seller else agent.awi.my_input_product]]
        negotiation_ranges = np.array(self._post_process_negotiation_ranges(negotiation_ranges)).flatten().tolist()
        last_offers = np.array(self._post_process_offers(last_offers)).flatten().tolist()

        if seller:
            result = np.concatenate((agent.current_time, last_offers, negotiation_ranges, agent.running_negotiations_count,
                                   agent.negotiation_requests_count,
                                   agent.contracts_count, price_product))
        else:
            result = np.concatenate((agent.current_time, last_offers, negotiation_ranges, agent.running_negotiations_count,
                                   agent.negotiation_requests_count,
                                   agent.contracts_count, price_product))

        logging.debug(f"Observation of {agent}'s {'seller' if seller else  'buyer'} are {result}")
        return result


    @staticmethod
    def _post_process_negotiation_ranges(negotiation_ranges):
        for index, nr in enumerate(negotiation_ranges):
            if nr:
                negotiation_ranges[index] = [list(issue.values) for issue in nr[0]]
            else:
                negotiation_ranges[index] = [[0, 0], [0, 0], [0, 0]]
        return negotiation_ranges

    @staticmethod
    def _post_process_offers(last_offers):
        for index, offer in enumerate(last_offers):
            if offer:
                last_offers[index] = np.array(offer).mean(axis=0).tolist()
            else:
                last_offers[index] = [0, 0, 0]
        return last_offers

    def reward(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        # sub-goal, best deal which is defined as being nearest to the agent needs with lowest price
        # main-goal, maximum profitability at the end of episode.
        # factory = world.a2f[agent.id]
        #print(f"{agent} balnce change is {factory.balance_change}")

        # delay reward
        rew = world.scores()[agent.id]

        # Timely reward
        _rew = np.mean(agent.reward)
        # if agent in world.policy_agents:
        #     print(f"{agent}\'s rewards are {agent.reward}")
        agent.reward = [0.0]

        if RANDOM_REWARD:
            return random.random()
        else:
            return rew + _rew

    def done(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return False

    def benchmark_data(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return {}