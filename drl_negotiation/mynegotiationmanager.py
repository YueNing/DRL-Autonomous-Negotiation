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
from typing import Tuple, List
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
        """
        #import ipdb
        #ipdb.set_trace()
        
        #print(f'{self}: negotiation manager {self.action.m}, issues{issues}')
        #if self.action.m in ([NegotiationRequestAction.ACCEPT_REQUEST], 
        #                        [NegotiationRequestAction.DEFAULT_REQUEST]):
        return self.negotiator(annotation["seller"] == self.id, issues=issues)
        #return None

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost
    
    def target_quantity(self, step: int, sell:bool) -> int:
        """
            Idea 4.1. TODO: observation: negotiations, negotiation_requests
                         action: target quantity, discrete action_space
            return target quantity

        """
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def _start_negotiations(
            self,
            product: int,
            sell: bool,
            step: int,
            qvalues: Tuple[int, int],
            uvalues: Tuple[int, int],
            tvalues: Tuple[int, int],
            partners: List[str] = None,
            ) -> None:
        """
            IDEA 4.3: TODO: observation: market conditions, target_quantity, 
                                            acceptable_unit_price, negotiations, negotiation_requests,
                                            qvalues, uvalues, tvalues, step, sell
                            action: range of issues
        """
        import numpy as np

        if not np.isin(self.action.m, 0).all() and sell:
            # set up observation
            # self.state.o_role = sell
            self.state.o_negotiation_step = self.awi.current_step
            #self.state.o_step = step
            #self.state.o_is_sell = sell

            #self.state.o_q_values = qvalues
            #self.state.o_u_values = uvalues
            #self.state.o_t_values = tvalues

            qvalues = tuple(np.array(qvalues) + (self.action.m[0:2] * (qvalues[1] - qvalues[0])).astype("int32"))
            uvalues = tuple(np.array(uvalues) + (self.action.m[2:4] * (uvalues[1] - uvalues[0])).astype("int32"))
            tvalues = tuple(np.array(tvalues) + (self.action.m[4:6] * (tvalues[1] - tvalues[0])).astype("int32"))
            #print(f"qvalues: {qvalues}, uvalues: {uvalues}, tvalues: {tvalues}")

        #import ipdb
        #ipdb.set_trace()
        #print(f"qvalues: {qvalues}, uvalues: {uvalues}, tvalues: {tvalues}")
        
        issues = [
                Issue(qvalues, name="quantity"),
                Issue(tvalues, name="time"),
                Issue(uvalues, name="uvalues")
                ]

        for partner in partners:
            self.awi.request_negotiation(
                    is_buy = not sell,
                    product = product,
                    quantity = qvalues,
                    unit_price = uvalues,
                    time = tvalues,
                    partner = partner,
                    negotiator = self.negotiator(sell, issues=issues)
                    )












