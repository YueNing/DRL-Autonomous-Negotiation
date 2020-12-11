'''
    Core class, functions
    Author: naodongbanana
    E-Mail: n1085633848@outlook.com
'''
import numpy as np
from scml.scml2020 import SCML2020World, SCML2020Agent

class AgentState:
    '''
        Agent state
    '''
    def __init__(self):
        # financial report
        # self.f = None
        # current step
        # self.c_step = None
        # management state, e.g. issues range
        self.m = None
        # communication utterance
        self.c = None

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
        # agent management action
        self.m = None
        # agent communication action, communication channel
        self.c = None

class MySCML2020Agent(SCML2020Agent):
    '''
        My scml 2020 agent, subclass of scml2020agent,
        action_callback: action decided by the callback
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # agents are manageable by default
        self.manageable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # management noise amount
        self.m_nois = None
        # communication noise amount
        self.c_nois = None
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # heuristic behavior to execute
        self.action_callback = None
        # agents are interactive
        self.interative = False

class TrainWorld(SCML2020World):
    """
    Multi-Agent, SCML world, used for training
    """

    def __init__(self, *args, **kwargs):
        # maddpg drived agents, heuristic agents, script drived agents, interative agents
        self.agents = []
        # SELLER, BUYER
        self.system_entities = []

        # communication channel dimensionality
        self.dim_c = 0
        # negotiation management dimensionality
        self.dim_m = 1
        # simulation timestep
        self.dt = 0.1

        super().__init__(*args, **kwargs)

    @property
    def entities(self):
        '''
            agents + system_entities
        '''
        return self.agents + self.system_entities

    @property
    def policy_agents(self):
        '''
           e.g. maddpg drived agents,
        '''
        return [agent for agent in self.agents if agent.action_callback is None]
    
    @property
    def heuristic_agents(self):
        '''
            e.g. script-drived agents
        '''
        return [agent for agent in self.agents if agent.action_callback is not None]

    @property
    def interactive_agents(self):
        '''
            e.g. controlled by user
        '''
        return [agent for agent in self.agents if agent.interactive]

    def step(self):
        # actions of policy agents are preset in environement.

        # set actions for heuristic_agents
        # controlled by scripts
        # agents have action_callback
        for agent in self.heuristic_agents:
            agent.action = agent.action_callback(agent, self)

        #TODO: set actions for interative_agents
        for agent in self.interactive_agents:
            # set by user
            pass

        super().step()
        
        # update agents' state
        for agent in self.agents:
            self.update_agent_state(agent)
    
    def update_agent_state(self, agent):
        # set management status
        if agent.blind:
            agent.state.m = np.zeros(self.dim_m)
        else:
            # TODO: get the management state
            agent.state.m = None
        
        # set communication status
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
