from drl_negotiation.scenario import BaseScenario
from drl_negotiation.core import TrainWorld
from drl_negotiation.myagent import MyComponentsBasedAgent
from scml.scml2020 import (
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
            SCML2020World,
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

        # configuration, for Scenario scml
        world_configuration = SCML2020World.generate(
            agent_types=agent_types,
            n_steps=n_steps
        )

        world = TrainWorld(configuration=world_configuration)

        self.reset_world(world)

        return world

    def reset_world(self, world):
        # callback, reset

        # reset world, agents, factories
        # fixed position
        agent_types = world.configuration['agent_types']
        agent_params = world.configuration['agent_params'][:-2]
        n_steps = world.configuration['n_steps']

        reset_configuration = SCML2020World.generate(
            #TODO: could be reset
            agent_types=agent_types,
            agent_params=agent_params,
            n_steps=n_steps
        )

        world.__init__(configuration=reset_configuration)

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # callback, reward
        # idea 1: external rewards, e.g. balance - initial balance for agent, -(balance - initial balance) for adversary agent
        # idea 2: Intrinsic motivation rewards.
        # On Learning Intrinsic Rewards for Policy Gradient Methods, https://arxiv.org/abs/1804.06459
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
        # callback: observation 
        
        # set financial goal

        #1. communication Economic gap with others
        economic_gaps = []
        for entity in world.entities:
            #if entity is agent: continue
            if entity in world.policy_agents:
                economic_gaps.append(np.array([entity.state.f]) - np.array([agent.state.f]))
            else:
                pass
        return np.concatenate(economic_gaps)

    def done(self, agent, world):
        # callback of done
        
        # simulation is end
        if world.world_done:
            return True

        import ipdb
        # agent is brankrupt
        return [_.is_bankrupt for _ in world.factories if _.agent_id == agent.id][0]
