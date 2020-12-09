"""
TODO: reconstruct the game as scenario
"""
from drl_negotiation.scenario import BaseScenario
from drl_negotiation.core import World
from drl_negotiation.negotiator import MyDRLNegotiator, MyOpponentNegotiator

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        world.agents = [MyDRLNegotiator(is_seller=False), MyOpponentNegotiator(is_seller=True)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'

    def reset_world(self):
        pass