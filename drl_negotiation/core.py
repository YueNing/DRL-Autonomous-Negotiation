from scml.scml2020 import SCML2020World, SCML2020Agent

class AgentState(object):
    def __init__(self):
        # financial report
        self.f = None
        # current step
        self.c_step = None
        # communication utterance
        self.c = None

class Action(object):
    def __init__(object):
        # negotiation action
        # accept or reject negotiation
        self.ac_n = None
        # communication action, communication channel
        self.ac_c = None

class MySCML2020Agent(SCML2020Agent):
    
    def __init__(self, *args, *kwargs):
        super().__init__(*args, **kwargs)

        # agents are negotiable by default
        self.negotiable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # negotiable noise amount
        # self.n_noise = None
        # communication noise amount
        self.c_nois = None
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # heuristic behavior to execute
        self.action_callback = None

class TrainWorld(SCML2020World):
    """
    Multi-Agent, SCML world
    """

    def __init__(self, *args, **kwargs):
        self.agents = []
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
        return self.agents + self.system_entities

    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]
    
    @property
    def heuristic_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    def step(self):
        
        # set actions for heuristic_agents
        # controlled by scripts
        for agent in self.heuristic_agents:
            agent.action = agent.action_callback(agent, self)

        super().step()

        for agent in self.agents:
            self.update_agent_state(agent)
    
    def update_agent_state(self, agent):
        # set negotiation state
        if not agent.observe:
            pass
        else:
            agent.state.f = self.get_financial_report(agent)
            agent.state.c_step = self.get_c_step(agent) 
        
        # set communication state
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
    
    def get_financial_report(self, agent):
        # TODO: return the financial report of agent
        pass

    def get_c_step(self, agent):
        # TODO: return the current step
        pass

