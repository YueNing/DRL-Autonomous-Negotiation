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
from scml_negotiation.myutilityfunction import MyUtilityFunction

# def my_utility_function(offer, time, rp, ip, t_end):
#     '''
#         TODO: need to use UtilityFunction from Negmas to define the utility of every issues
#         my utility function, 
#         consider both the outcome and time
#     '''
#     d_t = 0.6
#     # print(offer, time, ip, rp, t_end)
#     # print(((float(rp) - float(offer[0])) / (float(rp) - float(ip))) * (float(time) / float(t_end)) ** d_t)
#     return ((float(rp) - float(offer[0])) / (float(rp) - float(ip))) * (float(time) / float(t_end)) ** d_t

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
        return self._ufun
    
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
        self.set_env(env=env)

        # Must set it
        self._action: ResponseType = None
        self._proposal_offer = None
        

    def reset(self, env=None):
        if env:
            self.set_env(env=env)
        self._action = None
        self._proposal_offer = None

    @property
    def ufun(self):
        """
        Will remove it, 
        use the attribute _utility_function
        """
        return self._utility_function
    
    def get_obs(self) -> "outcome":
        '''
        Observation as,
        [opponent_Offer, Time] = [issue(quantity), issue(time), issue(unit price), time]
        '''
        return self.get_opponent_offer + [self.time]

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
    def get_opponent_offer(self):
        return self._opponent_offer
    
    @property
    def proposal_offer(self):
        return self._proposal_offer
    
    def set_proposal_offer(self, offer: "Outcome" = None):
        self._proposal_offer = offer
        
    def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
        """
        Remark:
            Use the action trained by drl model if it existes!
            Otherwise always reject the offer from opponent!
        """
        if self.action is None:
            if self.env is not None:
                return ResponseType(self.env.action_space.sample())
            
            return ResponseType.REJECT_OFFER
        
        if isinstance(self.action, ResponseType):
            return self.action
        
        return ResponseType.REJECT_OFFER

    # def propose(self, state: MechanismState) -> Optional["Outcome"]:
    #     '''
    #     when the counter has be called, to generate a new offer,
    #
    #     Remark:
    #
    #         Method0: implemented here
    #         heuristic method, design it by ourself.
    #
    #         Method1:
    #         From a machine learning perspective, deriving this proposal corresponds to a
    #         regression problem.
    #
    #         Method2 method:
    #         Or in reinforcement learning perspective, deriving this proposal corresponds to a
    #         continuous space problem.
    #         In other words: Work for negotiation, as the decision to accept or reject an offer is discrete,
    #         whereas bidding is on continuous space
    #
    #     '''
    #     assert self.action == ResponseType.REJECT_OFFER, "somethings error, action is not REJECT_OFFER, but go into function propose!"
    #
    #     # method0
    #     # get utility of all outcomes, proposal a new offer which utilitf is less biger that current  offer gived by the opponent
    #     # get it from ufun/_utility_function
    #
    #     #method1
    #     #need many data
    #
    #     #method2
    #     #a2c algorithm, need to design a a2c model contains two network(acceptance network, offer/bidding network)



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
        )

    @property
    def get_weights(self) -> List:
        return self._weights
    
        
    def propose(self):
        pass

    def response(self):
        pass


class MyOpponentNegotiator(DRLNegotiator, AspirationNegotiator):

    def __init__(
        self,
        name: str = 'my_opponent_negotiator', 
        ufun: Optional[UtilityFunction] = MappingUtilityFunction(lambda x: random.random() * x[0]),
    ):
        super().__init__(name=name)
        
        # Must set it
        self._ufun = ufun
    
    def get_obs(self):
        pass

    def get_current_action(self):
        pass
    
    def set_current_action(self, action=None):
        pass


#
# class MyNegotiator(SAONegotiator):
#     r"""
#     TODO:
#     """
#     def __init__(
#         self,
#         ufun: Optional[UtilityFunction] = None,
#     ):
#         super().__init__(name="my_negotiator")
#         self.current_action: Optional[ResponseType] = None
#         self.end_time = 1
#
#         # Parameters used for calcaulate utility value
#         self.initial_price = random.sample(range(300, 350), 1)[0]
#         self.reserved_price = random.sample(range(500, 550), 1)[0]
#
#         self.rational_proposal = True
#         self.random_proposal = False
#         # self.reserved_value = None
#
#     def propose_(self, state: MechanismState) -> Optional["Outcome"]:
#
#         if not self._ami._mechanism._running:
#             return None
#         proposal = self.propose(state=state)
#         # import pdb;pdb.set_trace()
#         # never return a proposal that is greater than the reserved value
#         if self.rational_proposal:
#             utility = None
#             if proposal is not None and self._utility_function is not None:
#                 utility = self.get_utility(proposal)
#                 if utility is not None and utility < self.get_utility((self.reserved_price,)):
#                     print('utility is less than the utility of reserved price')
#                     # when the utility of proposaled offer is less than the utility of reserved price,
#                     # then this proposaled offer is useless
#                     self.checked_proposal = False
#                     return None
#
#             if utility is not None:
#                 self.my_last_proposal = proposal
#                 self._my_last_proposal = proposal
#                 self.my_last_proposal_utility = utility
#                 self._my_last_proposal_utility = utility
#
#         return proposal
#
#     def respond(self, state: MechanismState, offer: "Outcome") -> "ResponseType":
#         print("The response of my negotiator has been called")
#         # if self.current_action == ResponseType.ACCEPT_OFFER:
#         #     import pdb;pdb.set_trace()
#         # print(self.current_action)
#         # if self.current_action == ResponseType.ACCEPT_OFFER:
#         #     import pdb;pdb.set_trace()
#
#         if isinstance(self.current_action, ResponseType):
#             return self.current_action
#
#     def propose(self, state: MechanismState) -> Optional["Outcome"]:
#         '''
#             Here define the function that how to propose,
#             just when the RepsonseType is REJECT_OFFER or the
#             offer from opponent is None, this function will be called!
#         '''
#         print("The propose of my negotiator has been called")
#         print(self.current_action)
#         # self.checked_proposal = False
#         if isinstance(self.current_action, ResponseType) \
#                 and self.current_action == ResponseType.REJECT_OFFER:
#             def _propose():
#                 '''
#                     TODO: calculate the new outcome
#                     initial: random propose,
#                     Improvement: the best outcome that has the best utility in current time
#                 '''
#                 if self.random_proposal:
#                     return self._ami.outcomes[random.randint(0, len(self._ami.outcomes))-1]
#                 else:
#                     # TODO: Continuous action space
#                     # add one value, this is a Continuous action space, return Gaussian probability distribution,
#                     # and then sample a value from this disctribution DDPG
#                     # return self._ami.outcomes[random.randint(0, len(self._ami.outcomes))-1]
#                     if self.my_last_proposal:
#                         print('my last proposal has been called!', self.my_last_proposal[0] + 5)
#                         return self.my_last_proposal[0] + 5,
#                     else:
#                         print('initial price')
#                         return self.initial_price + 1,
#
#             # print("_propose called")
#             self.checked_proposal = True
#             return _propose()
#         else:
#             # not offer received from opponent, so create random offer.
#             self.checked_proposal = True
#             return self._ami.outcomes[random.randint(0, len(self._ami.outcomes))-1]
#
#     def get_last_proposal(self):
#         return self.my_last_proposal
#
#     def get_current_action(self):
#         return self.current_action
#
#     def set_current_action(self, action: int):
#         # import pdb;pdb.set_trace()
#         # print("set current action called", ResponseType(action))
#         self.current_action = ResponseType(action)
#
#     def get_utility(self, offer: Optional["outcome"]):
#         return self.ufun(offer, self.get_time(), self.get_reserved_price(), \
#                     self.get_initial_price(), self.get_maximum_end_time())
#
#     def get_maximum_end_time(self):
#         '''
#             Get the end time of my negotiator
#         '''
#         if hasattr(self, 'end_time'):
#             return self.end_time
#         else:
#             return float("inf")
#
#     def get_initial_price(self):
#         '''
#             initial price which my negotiator offer
#         '''
#         return self.initial_price
#
#     def get_reserved_price(self):
#         '''
#             Maximum price which my negotiator can offer to opponent
#         '''
#         # import pdb;pdb.set_trace()
#         return self.reserved_price
#
#     def get_time(self):
#         # import pdb;pdb.set_trace()
#         # print(self._ami._mechanism.id)
#         # print(self._ami._mechanism.time)
#         return self._ami._mechanism.time
#
#     def get_obs(self, x_best) -> np.array:
#         '''
#         >>> get_obs(550)
#         np.array([550, 150.0, 335, 522])
#         '''
#         x_best = x_best
#         # import pdb;pdb.set_trace()
#         time_left = self.get_maximum_end_time() - self.get_time()
#         ip_my = self.get_initial_price()
#         rp_my = self.get_reserved_price()
#
#         result = [x_best, time_left, ip_my, rp_my]
#         scale_factory =  1 # scale inputs to be in the order of magnitude of 100 for neural network
#         return np.array(result) / scale_factory