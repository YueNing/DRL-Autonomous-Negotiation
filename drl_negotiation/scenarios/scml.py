from drl_negotiation.scenario import BaseScenario
from drl_negotiation.core import TrainWorld
from drl_negotiation.myagent import MyComponentsBasedAgent
from scml.scml2020 import (
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
            IndDecentralizingAgent,
            MovingRangeAgent
       )
import numpy as np

class Scenario(BaseScenario):

    def make_world(self):
        agent_types = [
                MyComponentsBasedAgent,
                DecentralizingAgent,
                BuyCheapSellExpensiveAgent
                ]
        n_steps = 10
        world = TrainWorld(agent_types=agent_types, n_steps=50)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        pass

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        #TODO:
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        #TODO: Difference from initial funds
        #agent.f - agent.init_f
        rew = 0
        return 0

    def adversary_reward(self, agent, world):
        #TODO: keep the good agents near the intial funds
        # neg reward
        # pos reward
        # agent.init_f - agent.f
        rew = 0
        return rew

    def observation(self, agent, world):
        #TODO? get all observation,
       
        # set financial goal

        #1. communication Economic gap with others
        economic_gaps = []
        for entity in world.entities:
            if entity is agent: continue
            if entity in world.policy_agents:
                economic_gaps.append(np.array([entity.state.f]) - np.array([agent.state.f]))
            else:
                pass
        return np.concatenate(economic_gaps+ [[1.0]])
