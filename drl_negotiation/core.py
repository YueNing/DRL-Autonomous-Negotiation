'''
    Core class, functions
    Author: naodongbanana
    E-Mail: n1085633848@outlook.com
'''
import numpy as np
from scml.scml2020 import SCML2020World, SCML2020Agent, is_system_agent
import  copy

class AgentState:
    '''
        Agent state
    '''
    def __init__(self):
        # physical position for rendering
        self.p_pos = (0, 0)
        # financial report
        self.f = 0
        # current step
        # self.c_step = None
        # management state, e.g. issues range
        self.m = None
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
        # agent management action
        self.m = None
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
        super().__init__(*args, **kwargs)
        # agents are manageable by default
        self.manageable = True
        # cannot send communication signals
        self.silent = True 
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
        # agents are adversary
        self.adversary = False

    def init(self):
        super(MySCML2020Agent, self).init()

class TrainWorld(SCML2020World):
    """
    Multi-Agent, SCML world, used for training
    """
    def __init__(self, configuration=None, *args, **kwargs):
        # maddpg drived agents, heuristic agents, script drived agents, interative agents
        # self.agents = []
        # SELLER, BUYER
        self.system_entities = []

        # communication channel dimensionality
        self.dim_c = 2
        # negotiation management dimensionality
        self.dim_m = 6 
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

        self.configuration = copy.deepcopy(configuration)

        super().__init__(**self.configuration)
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
        return [agent for agent in self.entities if agent.action_callback is None]
    
    @property
    def heuristic_agents(self):
        '''
            e.g. heuristic agents, BuyCheapSellExpensiveAgent
        '''
        return [agent for agent in self.entities if agent.action_callback=='heuristic']

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

        #TODO: simulation is already ends
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
            noise = np.random.randn(*agent.action.c.shape) * agent.c_nois if agent.c_nois else 0.0
            agent.state.c = agent.action.c + noise
