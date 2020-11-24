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
from typing import Optional, Union
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
from myutilityfunction import MyUtilityFunction, ANegmaUtilityFunction, MyOpponentUtilityFunction
from utils import normalize_observation, reverse_normalize_action

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
        # compare this time with the maximum_time, relative time
        return self._ami._mechanism.relative_time
    
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
        ufun: Union[UtilityFunction, MyUtilityFunction, ANegmaUtilityFunction, None] = None,
        env: "NEnv" = None,
        # presort = False,
        # For ANegma, single issue problem
        init_proposal = True,
        rp_range = None,
        ip_range = None,
        # model settings
        train: bool = True,
    ):
        super().__init__(
            name = name,
            ufun=ufun,
            # presort=presort,
        )
        # if env is not None:
        self.set_env(env=env)

        # Must set it
        self._action: ResponseType = None
        self._proposal_offer = None
        self._current_offer = None
        self.init_proposal = init_proposal
        if self.init_proposal:
            self._ip = None
            self._rp = None
            self._rp_range = rp_range
            self._ip_range = ip_range
        # if "ip" in ufun.__dict__ and ufun.__dict__['ip'] is not None:
        #     self._ip = self.ufun.ip

        self.train = train

    def reset(self, env=None):
        self._dissociate()
        if env:
            self.set_env(env=env)
        self._action = None
        self._proposal_offer = None
        self._current_offer_offer = None

        if self.init_proposal:
            self._ip = Issue.sample(issues=[Issue(self._ip_range)], n_outcomes=1, astype=tuple)[0]
            self._rp = Issue.sample(issues=[Issue(self._rp_range)], n_outcomes=1, astype=tuple)[0]

            if "ip" in self.ufun.__dict__:
                self.ufun.ip = self._ip
                self.set_proposal_offer(self._ip)

            if "rp" in self.ufun.__dict__:
                self.ufun.rp = self._rp

        self.end_time = 1


    @property
    def ufun(self):
        """
        Will remove it, 
        use the attribute _utility_function
        """
        return self._utility_function
    
    def get_obs(self) -> "outcome+time":
        '''
        Observation as, based on the design of observation space
        [opponent_Offer, Time] = [issue(quantity), issue(time), issue(unit price), time]

        TODO:
        Questions: how about initial observation when the current_offer is None

        Method1: random sample a outcome from issues,
        Method2: set all issues to default value, for example 0
        others?
        '''
        # if self.env.strategy == "ac_s":
        if self.get_current_offer is None:
            return self.reserved_obs()
        _obs = [i for i in self.get_current_offer] + [self.time]
        return normalize_observation(_obs, self)

    def reserved_obs(self):
        if self.env is not None:
            _obs = [0 for _ in self.env.game.issues] + [self.time]
            return normalize_observation(_obs, self)
        else:
            return None

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, action):
        self._action = ResponseType(action)

    def set_current_action(self, action=None):
        if self.env is not None:
            if self.env.strategy == "ac_s":
                self._action = ResponseType(action)
            elif self.env.strategy == "of_s":
                self._action = tuple(action)
        else:
            self._action = action

    def get_current_action(self):
        return self._action

    @property
    def get_current_offer(self):
        return self._current_offer

    def set_current_offer(self, offer):
        """

        Args:
            offer: in the side of negotiator,
                the offer proposed by opponents, different with the state.current_offer

        Returns:

        """
        self._current_offer = offer

    @property
    def proposal_offer(self):
        return self._proposal_offer
    
    def set_proposal_offer(self, offer: "Outcome" = None):
        # if "ip" in
        if "proposal_offer" in self.__dict__ and "ip" in self.ufun.__dict__:
            if self.proposal_offer is None and offer is not None and self.ufun.ip is None:
                self.ufun.ip = offer

        self._proposal_offer = offer
    
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        #print(f"\ncurrent offer of {self.name} is {offer}")
        action = super(DRLNegotiator, self).respond(state=state, offer=offer)
        #print(f"\033[1;32m response of {self.name} is {action} \033[0m")
        return action
    
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

        if self.init_proposal:
            self.init_proposal = False
            #print(f"\033[1;35m initial propose of {self.name} is {self.proposal_offer} \033[0m")
            return self.proposal_offer

        offer = super().propose(state=state)

        self.set_proposal_offer(offer)
        #print(f"\033[1;35m propose of {self.name} is {offer} \033[0m")

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
        env = None,
        # for Anegma, single issue,
        init_proposal:bool = True,
        rp_range = None,
        ip_range = None,
        # train model
        train: bool = True,
    ):
        """

        Args:
            name:
            is_seller:
            ufun:
            weights:
            env:
            init_proposal: initial proposal
            rp_range: if init_proposal is True, must set the rp_range
            ip_range: if init_proposal is True, must set the ip_range
            train: train or predict
        """
        
        if ufun is None:
            if is_seller:
                self._weights = weights[0]
            else: 
                self._weights = weights[1]
            
            ufun = MyUtilityFunction(weights=self.get_weights)

        super().__init__(
            name=name,
            ufun=ufun,
            env=env,
            init_proposal=init_proposal,
            rp_range=rp_range,
            ip_range=ip_range,
            train=train
        )

    @property
    def get_weights(self) -> List:
        return self._weights
        
    def propose(self, state: MechanismState) -> Optional["Outcome"]:
        if self.train:
            if "_env" in self.__dict__ and self.env is not None:
                if self.env.strategy == "ac_s":
                    return super().propose(state=state)
                elif self.env.strategy == "of_s":
                    if np.array(self.action) in self.env.action_space:
                        action = reverse_normalize_action(self.action, self)
                        # #print(f"\033[1;35m propose of {self.name} is {action} \033[0m")
                        return action
                        # return (540, )
                    else:
                        raise ValueError(f"The propose in {self.env.strategy} action {self.action} is error!")
                elif self.env.strategy == "hybrid":
                    raise NotImplementedError(f"The propose in {self.env.strategy} is not implemented!")
            else:
                return super(MyDRLNegotiator, self).propose(state=state)
        else:
            raise NotImplementedError(f"The propose of {self.name} in predict is not implemented!")

    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """
        Remark:
            Use the action trained by drl model if it existes!
            Otherwise always reject the offer from opponent!
        """
        #print(f"\ncurrent offer of {self.name} is {offer}")
        if self.train:
            self.set_current_offer(offer)
            if "_env" in self.__dict__ and self.env is not None:
                if self.env.strategy == "ac_s":
                    if self.action is None:
                        if self.env is not None:
                            action = ResponseType(self.env.action_space.sample())
                            #print(f"\033[1;32m response of {self.name} is {action} \033[0m")
                            return action
                        #print(f"\033[1;32m response of {self.name} is ResponseType.REJECT_OFFER! \033[0m")
                        return ResponseType.REJECT_OFFER

                    if isinstance(self.action, ResponseType):
                        #print(f"\033[1;32m response of {self.name} is {self.action} \033[0m!")
                        return self.action
                    #print(f"\033[1;32m response of {self.name} is ResponseType.REJECT_OFFER! \033[0m")
                    return ResponseType.REJECT_OFFER
                elif self.env.strategy == "of_s":
                    ##print(f"\033[1;32m response of {self.name} is ResponseType.REJECT_OFFER! \033[0m")
                    return ResponseType.REJECT_OFFER
                elif self.env.strategy == "hybrid":
                    raise NotImplementedError(f"The reponse of {self.name} in {self.env.strategy} is not implemented!")
                else:
                    raise ValueError(f"The reponse of {self.name} in {self.env.strategy} is illegal!")
            else:
                #print(f"\033[1;32m response of {self.name} is ResponseType.REJECT_OFFER! \033[0m")
                return ResponseType.REJECT_OFFER
        else:
            raise NotImplementedError(f"The reponse of {self.name} in predict is not implemented!")


class MyOpponentNegotiator(DRLNegotiator):

    def __init__(
        self,
        name: str = 'my_opponent_negotiator',
        is_seller: bool = True,
        ufun: Optional[UtilityFunction] = MyOpponentUtilityFunction,
        env = None
    ):
        if ufun is None:
            if is_seller:
                ufun = MyUtilityFunction(weights=(0, 0.25, 1))
            else:
                ufun = MyUtilityFunction(weights=(0, -0.5, -0.8))

        super().__init__(name=name, ufun=ufun, env=env, init_proposal=False)
    
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        self.set_current_offer(offer=offer)
        return super(MyOpponentNegotiator, self).respond(state=state, offer=offer)