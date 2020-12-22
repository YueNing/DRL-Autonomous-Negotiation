from drl_negotiation.scenario import BaseScenario
from drl_negotiation.core import TrainWorld, MySCML2020Agent
from drl_negotiation.myagent import MyComponentsBasedAgent
from scml.scml2020 import (
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
            SCML2020World,
       )
from typing import Union
import numpy as np

REW_FACTOR = 0.2

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
            #TODO: [Future work Improvement] could be reset
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
        # Delayed reward problem？？？？
        # Keep this in mind when writing reward functions: You get what you incentivize, not what you intend.
        # idea 1: external rewards, e.g. balance - initial balance for agent, -(balance - initial balance) for adversary agent
        # idea 2: Intrinsic motivation rewards.
        # On Learning Intrinsic Rewards for Policy Gradient Methods, https://arxiv.org/abs/1804.06459
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # 1. Difference of balance with the end of last step, begin of current step
        # TODO: 2. Difference of balance with the other agents
        rew = 0

        # means in this world step, the agent starts a sell negotiation except initial state
        if agent.state.o_negotiation_step == agent.awi.current_step:
            rew = (agent.state.f[2]- agent.state.f[1]) / (agent.state.f[0]) * REW_FACTOR
            agent.state.f[1] = agent.state.f[2]

        return rew

    def adversary_reward(self, agent, world):
        #TODO: keep the good agents near the intial funds
        # neg reward
        # pos reward
        # agent.init_f - agent.f
        rew = 0
        return rew

    def observation(self, agent: Union[MyComponentsBasedAgent, MySCML2020Agent], world: Union[TrainWorld]):
        # get all observation,
        # callback: obrvation

        # parameters needed to be observed
        # in order to determine the range of negotiation issues,

        # global information, e.g. market information

        # factory profile costs of every processes
        o_m = agent.awi.profile.costs

        # agent information, agent's
        o_a = np.array([agent._horizon])

        # unit_price
        #   1. catalog price in the market, not just input product and output product of agents.
        #   2. estimate of the current trading price of products from component Prediction Trading Strategy
        #   3. seller or buyer, seller or buyer are influenced by the outputs_needed, outputs_secured, inputs_needed
        #   and inputs_secured, Let iner learn this parameter by itself

        # catalog prices of products
        o_u_c = agent.awi.catalog_prices
        #TODO: excepted value after predict
        o_u_e = np.array([agent.expected_inputs, agent.expected_outputs, agent.input_cost, agent.output_price])
        #TODO: trading strategy, needed and secured
        o_u_t = np.array([agent.outputs_needed, agent.outputs_secured, agent.inputs_needed, agent.inputs_secured])

        # quantity
        #   1. seller or buyer
        #   2. target quantity
        #   3. running negotiation, negotiation request

        # running negotiation and negotiation request of agent
        o_q_n = np.array([
            agent.running_negotiations,
            agent.negotiation_requests,
        ])

        # time
        #   1. seller or buyer
        #   2. current step
        #   3. total step
        #   4. current step / total step
        o_t_c = np.array([agent.awi.current_step / agent.awi.n_steps])

        #2. Economic gap with others
        economic_gaps = []
        for entity in world.entities:
            if entity is agent: continue
            economic_gaps.append(entity.state.f - agent.state.f)
        economic_gaps = np.array(economic_gaps)

        #return np.concatenate(economic_gaps + o_m.flatten() + o_a + o_u_c + o_u_e + o_u_t + o_q_n.flatten() + o_t_c)
        return np.concatenate((economic_gaps.flatten(), o_m.flatten(), o_a, o_u_c, o_q_n.flatten(), o_t_c))

    def done(self, agent, world):
        # callback of done
        
        # simulation is end
        if world.world_done:
            return True

        import ipdb
        # agent is brankrupt
        return [_.is_bankrupt for _ in world.factories if _.agent_id == agent.id][0]