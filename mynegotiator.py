'''
DRL-Negotiator!
'''
#########################
## packages used for Test
from negmas import LinearUtilityFunction
#########################
from abc import ABC, abstractmethod, abstractproperty
import gym

import numpy as np
from typing import Optional
from negmas import Action #  An action that an `Agent` can execute in a `World` through the `Simulator`
from negmas import (
                        Issue, 
                        MechanismState, # The mechanism state
                        SAOMechanism, 
                        SAONegotiator,  
                        MappingUtilityFunction, 
                        AspirationNegotiator,
                        ResponseType,
                        UtilityFunction,
                        UtilityValue,
                        AgentMechanismInterface,
                        Outcome,
                        outcome_for,
                        outcome_as_tuple,
                    )
    
import random
from typing import List, Optional, Type, Sequence
from myutilityfunction import MyUtilityFunction

__all__ = [
    "DRLMixIn",
    "CommonMixIn",
    "DRLNegotiator",
    "MyDRLNegotiator",
    "MyOpponentNegotiator"
]

class DRLMixIn(ABC):
    '''
    TODO:
    Define the functions of agent that can used drl method to train itself!!
    '''
    
    @abstractmethod
    def get_obs(self)-> gym.spaces:
        raise NotImplementedError("Error: function get_obs has not been implemented!")
    
    @abstractmethod
    def get_current_action(self):
        raise NotImplementedError("Error: function get_current_action has not been implemented!")

    @abstractmethod
    def set_current_action(self, action=None):
        raise NotImplementedError("")


class CommonMixIn:
    
    @property
    def get_ufun(self):
        return self.ufun
    
    @property
    def time(self):
        return self._ami._mechanism.time
    
    @property
    def maximum_time(self):
        if hasattr(self, 'end_time'):
            return self.end_time
        
        return float("inf")
    
    def set_env(self, env: "NEnv" = None):
        self._env = env
    
    @property
    def env(self):
        return self._env

class DRLNegotiator(DRLMixIn, CommonMixIn, AspirationNegotiator):
    """
        Base class for all negotiator which want to join the drl game, saonegotiator,
        Common deep reinforcement learning negotiator.

        In this case just implements
            Acceptance strategy with drl method,
            Offer/bidding strategy with the propose function defined in aspiration negotiator

        args:
        (aei: agent environment interface)

        abstract function:
            get_current_action
            proposal
            respond
    
    """

    def __init__(
        self, 
        name: str = 'drl_negotiator',
        ufun: Optional[UtilityFunction] = None,
        env: "NEnv" = None, 
    ):
        super().__init__(
            name = name,
            ufun=ufun
        )
        if env is not None:
            self.set_env(env=env)

        # Must set it
        self._action: ResponseType = None
        self._proposal_offer = None
        self._current_offer = None
        

    def reset(self, env=None):
        if env:
            self.set_env(env=env)
        self._action = None
        self._proposal_offer = None
        self._current_offer_offer = None

    @property
    def ufun(self):
        """
        Will remove it, 
        use the attribute _utility_function
        """
        return self._utility_function
    
    def get_obs(self) -> "outcome+time":
        '''
        Observation as,
        [opponent_Offer, Time] = [issue(quantity), issue(time), issue(unit price), time]

        TODO:
        Questions: how about initial observation when the current_offer is None

        Method1: random sample a outcome from issues,
        Method2: set all issues to default value, for example 0
        others?
        '''
        if self.get_current_offer is None:
            return self.reserved_obs()
        return [i for i in self.get_current_offer] + [self.time]

    def reserved_obs(self):
        return [0, 0, 0] + [self.time]

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action):
        self._action = ResponseType(action)

    def set_current_action(self, action=None):
        self._action = ResponseType(action)

    def get_current_action(self):
        return self._action

    @property
    def get_current_offer(self):
        return self._current_offer

    def set_current_offer(self, offer):
        self._current_offer = offer

    @property
    def proposal_offer(self):
        return self._proposal_offer
    
    def set_proposal_offer(self, offer: "Outcome" = None):
        self._proposal_offer = offer
    
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        return super(DRLNegotiator, self).respond(state=state, offer=offer)
    
    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        '''
        when the counter has be called, to generate a new offer,

        Remark:

            Method0: implemented here
            heuristic method, design it by ourself.

            Method1:
            From a machine learning perspective, deriving this proposal corresponds to a
            regression problem.

            Method2 method:
            Or in reinforcement learning perspective, deriving this proposal corresponds to a
            continuous space problem.
            In other words: Work for negotiation, as the decision to accept or reject an offer is discrete,
            whereas bidding is on continuous space

        '''
        # assert self.action == ResponseType.REJECT_OFFER, "somethings error, action is not REJECT_OFFER, but go into function propose!"

        offer = super().propose(state=state)

        self.set_proposal_offer(offer)

        return offer

        # method0
        # get utility of all outcomes, proposal a new offer which utilitf is less biger that current  offer gived by the opponent
        # get it from ufun/_utility_function

        #method1
        #need many data

        #method2
        #a2c algorithm, need to design a a2c model contains two network(acceptance network, offer/bidding network)



class MyDRLNegotiator(DRLNegotiator):
    '''
    Negotiator which combines both Acceptance and Offer strategy

    Default issues quantity, time, unit_price,
    based on the settings of negotiation in scml.
    for example need to consider the Negotiator as seller or buyer,
    different role of Negotiator will use different weights in utility function.
    '''
    def __init__(
        self,
        name: str = "my_drl_negotiator",
        is_seller: bool = True,
        ufun: Optional[UtilityFunction] = None,
        weights: Optional[List[List]] = [(0, 0.25, 1), (0, -0.5, -0.8)],
        env = None
    ):
        
        if ufun is None:
            if is_seller:
                self._weights = weights[0]
            else: 
                self._weights = weights[1]
            
            ufun = MyUtilityFunction(weights=self.get_weights)

        super().__init__(
            name=name,
            ufun=ufun,
            env=env
        )

    @property
    def get_weights(self) -> List:
        return self._weights
    
        
    # def propose(self):
    #     pass

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """
        Remark:
            Use the action trained by drl model if it existes!
            Otherwise always reject the offer from opponent!
        """
        self.set_current_offer(offer)
        if self.action is None:
            if self.env is not None:
                return ResponseType(self.env.action_space.sample())

            return ResponseType.REJECT_OFFER

        if isinstance(self.action, ResponseType):
            return self.action

        return ResponseType.REJECT_OFFER


class MyOpponentNegotiator(DRLNegotiator):

    def __init__(
        self,
        name: str = 'my_opponent_negotiator',
        is_seller: bool = True,
        ufun: Optional[UtilityFunction] = MappingUtilityFunction(lambda x: random.random() * x[0]),
        env = None
    ):
        if ufun is None:
            if is_seller:
                ufun = MyUtilityFunction(weights=(0, 0.25, 1))
            else:
                ufun = MyUtilityFunction(weights=(0, -0.5, -0.8))

        super().__init__(name=name, ufun=ufun, env=env)
    
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        self.set_current_offer(offer=offer)
        super(MyOpponentNegotiator, self).respond(state=state, offer=offer)