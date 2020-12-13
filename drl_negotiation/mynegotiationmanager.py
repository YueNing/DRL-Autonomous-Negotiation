'''
   negotiation manager, for scml agent
   Author: naodongbanana
   E-Mail: n1085633848@outlook.com
'''
from typing import List, Dict, Optional, Any

from scml import NegotiationManager
################ For Test #### will be removed #########
from scml.scml2020 import IndependentNegotiationsManager
########################################################
from negmas import AgentMechanismInterface, Negotiator, Issue, SAONegotiator
import numpy as np

from .negotiator import MyDRLNegotiator
from .core import NegotiationRequestAction

class MyNegotiationManager(IndependentNegotiationsManager):
    """
        my negotiation manager
    """
    def respond_to_negotiation_request(
            self,
            initiator: str,
            issues: List[Issue],
            annotation: Dict[str, Any],
            mechanism: AgentMechanismInterface,
            ) -> Optional[Negotiator]:
        """
            IDEA 4.2: TODO: observation: finanical report of initiator
                            action: ACCEPT or REJECT to negotiate
            IDEA 4.3: TODO: observation: market conditions
                            action: range of issues
        """ 
        print(f'{self}: negotiation manager {self.action.m}')
        if self.action.m in ([NegotiationRequestAction.ACCEPT_REQUEST], 
                                [NegotiationRequestAction.DEFAULT_REQUEST]):
            return self.negotiator(annotation["seller"] == self.id, issues=issues)
        return None

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost
    
    def target_quantity(self, step: int, sell:bool) -> int:
        """
            Idea 4.1. TODO: observation: negotiations, negotiation_requests
                         action: target quantity, continuous action_space
            return target quantity

        """
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]
