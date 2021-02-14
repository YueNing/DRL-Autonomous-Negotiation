'''
    Core class, functions
    Author: naodongbanana
    E-Mail: n1085633848@outlook.com
'''
import numpy as np
from scml.scml2020 import SCML2020World, SCML2020Agent, is_system_agent
from typing import Optional
from drl_negotiation.core.hyperparameters import *
import yaml
import  copy
import pickle

class AgentState:
    '''
        Agent state
    '''
    def __init__(self):
        # physical position for rendering
        self.p_pos = (0, 0)
        # others state
        self.o_negotiation_step = 0
        # financial report
        self.f: np.array = np.zeros(3)
        # self.f_init = 0
        # self.f_begin = 0
        # self.f_end = 0
        # current step
        # self.o_current_step = 0
        # management state, e.g. issues range
        # self.m = None
        # communication utterance
        self.c = None

class NegotiationRequestAction:
    DEFAULT_REQUEST = 0.0
    ACCEPT_REQUEST = 1.0
    REJECT_REQUEST = -1.0

class Action:
    '''
        agent's action
        m: management action
            e.g. discrete action --- accept or reject negotiation request
                 continuous action --- range of issues for negotiating,
                 (min, max, min, max, min, max)
        c: communication action
            e.g. send the info into public channel, secured, needs, negotiations, requests, 
                or info of competitors predicted by agent
    '''
    def __init__(self):
        # agent management action, used after training, in test periode
        self.s = None
        self.s_vel = None

        # seller, used in training
        self.m = None
        self.m_vel = 5
        # buyer, used in training
        self.b = None
        self.b_vel = 3

        # agent communication action, communication channel
        self.c = None

class MySCML2020Agent(SCML2020Agent):
    '''
        My scml 2020 agent, subclass of scml2020agent,
        action_callback: action decided by the callback

        hook:
            init
    '''
    Owner = 'My'

    def __init__(self, *args, **kwargs):
        # agents are adversary
        self.adversary = kwargs.pop("adversary") if "adversary" in kwargs else False
        # agents are manageable by default
        self.manageable = MANAGEABLE
        # cannot send communication signals
        self.silent = SLIENT
        # cannot observe the world
        self.blind = BLIND
        # management noise amount
        self.m_nois = None
        # communication noise amount
        self.c_nois = None
        # manageable range
        self.m_range = 1.0
        self.b_range = 1.0

        # reward
        self.reward = [0.0]

        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # heuristic behavior to execute
        self.action_callback = None
        # agents are interactive
        self.interative = False
        super().__init__(*args, **kwargs)

    def init(self):
        super(MySCML2020Agent, self).init()

    @property
    def running_negotiations_count(self) -> [int, int]:
        """

        Returns:
            number of runniing negotiations (sell, buy)
        """

        return self._count(super(MySCML2020Agent, self).running_negotiations)


    @property
    def negotiation_requests_count(self) -> [int, int]:
        """

        Returns:
            number of standing negotiation requests, sell, buy
        """
        return self._count(super(MySCML2020Agent, self).negotiation_requests)

    @property
    def contracts_count(self):
        number_buy_contracts = 0
        number_sell_contracts = 0
        if self.contracts:
            for c in self.contracts:
                if c.annotation['buyer'] == self.id:
                    number_buy_contracts += 1
                if c.annotation['seller'] == self.id:
                    number_sell_contracts += 1
        return number_buy_contracts, number_sell_contracts

    @property
    def current_time(self):
        return [self.awi.current_step / self.awi.n_steps]

    def _count(self, negotiations):
        sell = 0
        buy = 0
        for n in negotiations:
            if n.annotation["seller"] == self.id:
                sell +=1
            elif n.annotation["buyer"] == self.id:
                buy +=1
        return sell, buy
      
    def _get_obs(self, seller=True, scenario="scml"):
        # local observation
        return

    def init(self):
        super(MySCML2020Agent, self).init()
        if RUNNING_IN_SCML2020World:
            if not self.train:
                self._setup_model()



class TrainWorld(SCML2020World):
    """
    Multi-Agent, SCML world, used for training
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
        self.dim_m = DIM_M # seller
        self.dim_b = DIM_B # buyer

        #TODO: limits on the negotiation agenda size
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

                #TODO: interactive test
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
        dump_data_yaml = {"agent_types": [_._type_name() for _ in self.configuration['agent_types']],
                    'agent_params': self.configuration['agent_params'],
                    "n_steps": self.n_steps,
                    "negotiation_speed": self.configuration["negotiation_speed"],
                    "process_inputs": self.configuration["process_inputs"].tolist(),
                    "process_outputs": self.configuration["process_outputs"].tolist(),
                    "catalog_prices": self.configuration["catalog_prices"].tolist(),
                    # "profiles": [profile.costs.tolist() for profile in self.profiles],
                    # "exogenous_contracts": self.exogenous_contracts,
                    # "info": self.info,
                    "force_signing": self.configuration["force_signing"],
                    "exogenous_horizon": self.configuration["exogenous_horizon"]
                     }

        with open(file_name+'.yaml', "w") as file:
            yaml.safe_dump(dump_data_yaml, file)

        logging.info(f"{file_name}.yaml saved")

        dump_data_pkl = self.configuration
        with open(file_name+'.pkl', 'wb') as file:
            pickle.dump(dump_data_pkl, file)

        logging.info(f"{file_name}.pkl saved")
        # super().save_config(file_name=file_name)