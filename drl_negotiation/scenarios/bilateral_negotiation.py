"""
TODO: reconstruct the game as scenario
"""
from drl_negotiation.scenarios.scenario import BaseScenario
from drl_negotiation.core.core import TrainWorld
from drl_negotiation.agents.negotiator import MyDRLNegotiator, MyOpponentNegotiator

class Scenario(BaseScenario):
    def make_world(self):
        world = TrainWorld()
        world.agents = [MyDRLNegotiator(is_seller=False), MyOpponentNegotiator(is_seller=True)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'

    def reset_world(self):
        pass