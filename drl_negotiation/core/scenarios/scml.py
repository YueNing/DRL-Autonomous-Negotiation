from drl_negotiation.core.scenarios.scenario import BaseScenario
from drl_negotiation.core.games.scml import TrainWorld, MySCML2020Agent
from drl_negotiation.agents.myagent import MyComponentsBasedAgent
from drl_negotiation.core.config.hyperparameters import *
from drl_negotiation.third_party.negmas.negmas.helpers import get_class
from drl_negotiation.third_party.scml.src.scml.scml2020 import (
            DecentralizingAgent,
            BuyCheapSellExpensiveAgent,
            SCML2020World,
            is_system_agent,
       )
from typing import Union
import numpy as np

class Scenario(BaseScenario):

    def make_world(self, config=None) -> TrainWorld:
        # configuration, for Scenario scml
        if config is None:
            agent_types = [get_class(agent_type, ) for agent_type in TRAINING_AGENT_TYPES]
            n_steps = N_STEPS
            world_configuration = SCML2020World.generate(
                agent_types=agent_types,
                n_steps=n_steps
            )
        else:
            world_configuration = SCML2020World.generate(
                agent_types=config['agent_types'],
                agent_params=config['agent_params'][:-2],
                n_steps=config['n_steps']
            )

        world = TrainWorld(configuration=world_configuration)

        if config is None:
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

    def benchmark_data(self, agent, world, seller=True):
        #TODO: data for benchmarkign purposes, info_callabck,
        # will be rendered when display is true
        # how to compare different companies, Ratio Analysis
        # https://www.investopedia.com/ask/answers/032315/how-does-ratio-analysis-make-it-easier-compare-different-companies.asp
        # price-to-earnings ratio and net profit margin
        # Margin Ratios and  Return Ratios
        # https://corporatefinanceinstitute.com/resources/knowledge/finance/profitability-ratios/
        profitability = []
        initial_balances = []
        factories = [_ for _ in world.factories if not is_system_agent(_.agent_id)]
        for i, factory in enumerate(factories):
            initial_balances.append(factory.initial_balance)
        normalize = all(_ != 0 for _ in initial_balances)

        for _ in world.agents:
            if world.agents[_].action_callback == "system": continue
            if world.agents[_] in world.heuristic_agents:
                if normalize:
                    profitability.append(
                    (agent.state.f[2] - agent.state.f[0]) / agent.state.f[0] -
                    ([f.current_balance for f in factories if f.agent_id == world.agents[_].id][0] -
                     [f.initial_balance for f in factories if f.agent_id == world.agents[_].id][0]) /
                    [f.initial_balance for f in factories if f.agent_id == world.agents[_].id][0]
                )
                else:
                    profitability.append(
                        (agent.state.f[2] - agent.state.f[0]) -
                        ([f.current_balance for f in factories if f.agent_id == world.agents[_].id][0] -
                         [f.initial_balance for f in factories if f.agent_id == world.agents[_].id][0])
                    )

        return {"profitability": profitability}

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world, seller=True):
        # callback, reward
        # Delayed reward problem？？？？
        # Keep this in mind when writing reward functions: You get what you incentivize, not what you intend.
        # idea 1: external rewards, e.g. balance - initial balance for agent, -(balance - initial balance) for adversary agent
        # idea 2: Intrinsic motivation rewards.
        # On Learning Intrinsic Rewards for Policy Gradient Methods, https://arxiv.org/abs/1804.06459
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # 1. Difference of balance with the end of last step, begin of current step
        # 2. Difference of balance with the other agents
        rew = 0

        # means in this world step, the agent starts a sell negotiation except initial state
        if agent.state.o_negotiation_step == agent.awi.current_step:
            rew = (agent.state.f[2]- agent.state.f[1]) / (agent.state.f[0]) * REW_FACTOR

        gap = []
        for entity in world.entities:
            if entity is agent: continue
            if entity.action_callback == "system": continue
            if entity.action_callback is None: continue
            initial_balance = [_.initial_balance for _ in world.factories if _.agent_id == entity.id][0]
            current_balance = [_.current_balance for _ in world.factories if _.agent_id == entity.id][0]
            gap.append((current_balance - initial_balance) / initial_balance)

        rew -= np.mean(np.array(gap))
        return rew

    def adversary_reward(self, agent, world):
        #TODO: keep the good agents near the intial funds
        # neg reward
        # pos reward
        # agent.init_f - agent.f
        rew = 0
        return rew

    def observation(self, agent: Union[MyComponentsBasedAgent, MySCML2020Agent], world: Union[TrainWorld], seller=True):

        o_m = agent.awi.profile.costs
        o_m = o_m[:, agent.awi.profile.processes]

        # agent information, agent's
        o_a = np.array([agent._horizon])

        # catalog prices of products
        o_u_c = agent.awi.catalog_prices
        # TODO: excepted value after predict
        o_u_e = np.array([agent.expected_inputs, agent.expected_outputs, agent.input_cost, agent.output_price])
        # TODO: trading strategy, needed and secured
        o_u_t = np.array([agent.outputs_needed, agent.outputs_secured, agent.inputs_needed, agent.inputs_secured])

        # running negotiation and negotiation request of agent
        o_q_n = np.array([
            agent.running_negotiations_count,
            agent.negotiation_requests_count,
        ])

        o_t_c = np.array([agent.awi.current_step / agent.awi.n_steps])

        # 2. Economic gap
        economic_gaps = []
        economic_gaps.append(agent.state.f[2] - agent.state.f[1])
        economic_gaps = np.array(economic_gaps)

        # return np.concatenate(economic_gaps + o_m.flatten() + o_a + o_u_c + o_u_e + o_u_t + o_q_n.flatten() + o_t_c)

        return np.concatenate((economic_gaps.flatten(), o_m.flatten(), o_a, o_u_c, o_q_n.flatten(), o_t_c))

    def done(self, agent, world, seller=True):
        # callback of done
        
        # simulation is end
        if world.world_done:
            return True

        import ipdb
        # agent is brankrupt
        return [_.is_bankrupt for _ in world.factories if _.agent_id == agent.id][0]