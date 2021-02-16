import yaml
import copy
import pickle
import numpy as np
import logging
from typing import Optional
from drl_negotiation.third_party.scml.src.scml.scml2020 import SCML2020World, is_system_agent
from drl_negotiation.core.config.hyperparameters import (COLLABORATIVE, DIM_M, DIM_B, QUANTITY, TIME, UNIT_PRICE)
from drl_negotiation.core.games._scml import AgentState, MySCML2020Agent

__all__ = [
    "TrainWorld"
]


class TrainWorld(SCML2020World):
    """
    Multi-Agent, SCML world, used for training,
    standard SCML2020World
    """

    def __init__(self, configuration=None, *args, **kwargs):
        # maddpg drived agents, heuristic agents, script drived agents, interative agents
        # self.agents = []
        # SELLER, BUYER
        self.collaborative = COLLABORATIVE
        self.system_entities = []

        # communication channel dimensionality
        self.dim_c = 2
        # negotiation management dimensionality
        self.dim_m = DIM_M  # seller
        self.dim_b = DIM_B  # buyer

        # TODO: limits on the negotiation agenda size
        self.Q = QUANTITY
        self.T = TIME
        self.U = UNIT_PRICE

        # simulation timestep
        self.dt = 0.1

        # world done
        self.__done = False

        # set up the scml2020world
        if configuration is None:
            configuration = SCML2020World.generate(
                *args,
                **kwargs
            )

        self._configuration = copy.deepcopy(configuration)
        self._configuration['no_logs'] = True
        # backup for reset
        self.configuration = copy.deepcopy(self._configuration)

        super().__init__(**self._configuration)
        # set action_callback for agent which hasnot it
        for agent in self.agents.values():
            if not hasattr(agent, 'action_callback'):
                if is_system_agent(agent.id):
                    agent.action_callback = 'system'
                    self.system_entities.append(agent)
                else:
                    agent.action_callback = 'heuristic'

            if not hasattr(agent, 'interactive'):
                agent.interactive = False

            if not hasattr(agent, 'state'):
                agent.state = AgentState()

    @property
    def entities(self):
        '''
            agents + system_entities
        '''
        return [agent for agent in self.agents.values()]

    @property
    def policy_agents(self):
        '''
           e.g. maddpg drived agents,
        '''
        return [agent for agent in self.entities if agent.action_callback is None and not agent.adversary]

    @property
    def heuristic_agents(self):
        '''
            e.g. heuristic agents, BuyCheapSellExpensiveAgent
        '''
        return [agent for agent in self.entities if agent.action_callback is None and agent.adversary]

    @property
    def interactive_agents(self):
        '''
            e.g. controlled by user
        '''
        return [agent for agent in self.entities if agent.interactive]

    @property
    def script_agents(self):
        '''
            My script-drived agents, with action_callback
        '''
        return [agent for agent in self.entities if callable(agent.action_callback)]

    def step(self):
        # actions of policy agents are preset in environement.

        # set actions for heuristic_agents
        # controlled by scripts
        # agents have action_callback
        for agent in self.script_agents:
            agent.action = agent.action_callback(agent, self)

        # simulation is already ends
        if self.time >= self.time_limit:
            self.__done = True
            return

        if not super().step():
            self.__done = True
            return

            # update agents' state
        # policy agents
        for agent in self.policy_agents:
            self.update_agent_state(agent)

    @property
    def world_done(self):
        '''
            running info of world
        '''
        return self.__done

    def update_agent_state(self, agent: Optional[MySCML2020Agent]):
        # initial update the state of
        if agent.awi.current_step == 0:
            f_init = [_.initial_balance for _ in self.factories if _.agent_id == agent.id][0]
            f_begin = f_init
            f_end = f_begin
            agent.state.f = np.array([f_init, f_begin, f_end])
        else:
            # set financial status
            if agent.blind:
                # agent.state.m = np.zeros(self.dim_m)
                agent.state.f = np.zeros(3)
            else:

                # update agent state, get the management state
                # qvalues = (1, agent.target_quantity(agent.state.o_step, agent.state.o_is_sell))
                # tvalues = agent._trange(agent.state.o_negotiation_step, agent.state.o_step)
                # uvalues = agent._urange(agent.state.o_step, agent.state.o_is_sell, tvalues)
                # agent.state.m = [qvalues, tvalues, uvalues]

                f_end = [_.current_balance for _ in self.factories if _.agent_id == agent.id][0]
                agent.state.f[2] = f_end

                # TODO: interactive test
                agent.state.o_negotiation_step = agent.awi.current_step

                if agent.state.o_negotiation_step == agent.awi.current_step:
                    # after calculate the reward, then update the f_begin
                    pass
                else:
                    f_begin = f_end
                    agent.state.f[1] = f_begin

            # set communication status
            if agent.silent:
                agent.state.c = np.zeros(self.dim_c)
            else:
                noise = np.random.randn(*agent.action.c.shape) * agent.c_nois if agent.c_nois else 0.0
                agent.state.c = agent.action.c + noise

    def save_config(self, file_name: str):
        """
            dict generated by the SCML2020World.generate
            return dict(
            process_inputs=process_inputs,
            process_outputs=process_outputs,
            catalog_prices=catalog_prices,
            profiles=[_[0] for _ in profile_info],
            exogenous_contracts=exogenous,
            agent_types=agent_types,
            agent_params=agent_params,
            initial_balance=initial_balance,
            n_steps=n_steps,
            info=info,
            force_signing=force_signing,
            exogenous_horizon=horizon,
            **kwargs,
        )
        """
        dump_data_yaml = {"agent_types"      : [_._type_name() for _ in self.configuration['agent_types']],
                          'agent_params'     : self.configuration['agent_params'],
                          "n_steps"          : self.n_steps,
                          "negotiation_speed": self.configuration["negotiation_speed"],
                          "process_inputs"   : self.configuration["process_inputs"].tolist(),
                          "process_outputs"  : self.configuration["process_outputs"].tolist(),
                          "catalog_prices"   : self.configuration["catalog_prices"].tolist(),
                          # "profiles": [profile.costs.tolist() for profile in self.profiles],
                          # "exogenous_contracts": self.exogenous_contracts,
                          # "info": self.info,
                          "force_signing"    : self.configuration["force_signing"],
                          "exogenous_horizon": self.configuration["exogenous_horizon"]
                          }

        with open(file_name + '.yaml', "w") as file:
            yaml.safe_dump(dump_data_yaml, file)

        logging.info(f"{file_name}.yaml saved")

        dump_data_pkl = self.configuration
        with open(file_name + '.pkl', 'wb') as file:
            pickle.dump(dump_data_pkl, file)

        logging.info(f"{file_name}.pkl saved")
        # super().save_config(file_name=file_name)