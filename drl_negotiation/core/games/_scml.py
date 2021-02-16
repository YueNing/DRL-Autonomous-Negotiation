"""
    Core class, functions
    Author: naodongbanana
    E-Mail: n1085633848@outlook.com
"""
import numpy as np
from scml.scml2020 import SCML2020Agent
from drl_negotiation.core.config.hyperparameters import (MANAGEABLE, SLIENT, BLIND,
                                                         RUNNING_IN_SCML2020World
                                                         )

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
                sell += 1
            elif n.annotation["buyer"] == self.id:
                buy += 1
        return sell, buy

    def _get_obs(self, seller=True, scenario="scml"):
        # local observation
        return

    def init(self):
        super(MySCML2020Agent, self).init()
        if RUNNING_IN_SCML2020World:
            if not self.train:
                self._setup_model()