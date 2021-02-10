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

        for negotiation in agent.running_negotiations:
            logging.debug(f"{agent}:{negotiation.annotation}")

        if hasattr(agent, "controllers"):
            for is_seller, controller in agent.controllers.items():
                for nid in controller.history_offers:
                    negotiation = [negotiation for negotiation in controller.history_running_negotiations if negotiation.negotiator == controller.negotiators[nid][0]]
                    if negotiation:
                        if negotiation[0].annotation["seller"] == agent.id:
                            partner = negotiation[0].annotation["buyer"]
                        else:
                            partner = negotiation[0].annotation["seller"]

                        last_offers[sorted(agent.awi.my_consumers + agent.awi.my_suppliers).index(partner)].append(controller.history_offers[nid])

                controller.history_offers = {}

        price_product = [agent.awi.catalog_prices[agent.awi.my_output_product if seller else agent.awi.my_input_product]]
        last_offers = np.array(self._post_process_offers(last_offers)).flatten().tolist()
        logging.debug(f"{agent}'s {'seller' if seller else  'buyer'} last_offers {last_offers}")

        if seller:
            return np.concatenate((agent.current_time, last_offers, agent.running_negotiations_count,
                                   agent.negotiation_requests_count,
                                   agent.contracts_count, price_product))
        else:
            return np.concatenate((agent.current_time, last_offers, agent.running_negotiations_count,
                                   agent.negotiation_requests_count,
                                   agent.contracts_count, price_product))

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
        factory = world.a2f[agent.id]
        #print(f"{agent} balnce change is {factory.balance_change}")
        rew = world.scores()[agent.id]
        if RANDOM_REWARD:
            return random.random()
        else:
            return rew

    def done(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return False

    def benchmark_data(self, agent: MySCML2020Agent, world: TrainWorld, seller=True):
        return {}