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

class MyNegotiationManager(IndependentNegotiationsManager):
    """
        my negotiation manager
    """
    
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost
    
    def target_quantity(self, step: int, sell:bool) -> int:

        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]
