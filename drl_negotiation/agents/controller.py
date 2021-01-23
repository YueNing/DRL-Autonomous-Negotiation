"""
DRL Controller
"""
############################
# Packages used for Test
############################
import logging
import random
from typing import Optional, Dict, Tuple
from negmas import (Outcome,
                    SAOResponse,
                    SAOState,
                    MechanismState,
                    ResponseType,
                    )

from scml.scml2020.services import SyncController
import numpy as np
from scml.scml2020.common import UNIT_PRICE
from negmas import ResponseType
from drl_negotiation.core.hyperparameters import RANDOM


##########################################################################################################
# Controller for SCML, Used for training concurrent negotiation with DRL
# Author naodongbanana
# Datum 25.12.2020
##########################################################################################################
class MyDRLSCMLSAOSyncController(SyncController):
    """
    TODO:
    A Controller that used by Deep Reinforcement learning method, can manage multiple negotiators synchronously
    reward of current step,
    reward of whole simulation step,
    will try to let the factory/agent get the maximum profitability at the end
    Args:
    """

    def __init__(self,
                 parent: "PredictionBasedTradingStrategy",
                 is_seller: bool = None,
                 **kwargs
                 ):
        super().__init__(
            is_seller=is_seller,
            parent=parent,
            price_weight=kwargs.pop('price_weight'),
            utility_threshold=kwargs.pop('utility_threshold'),
            time_threshold=kwargs.pop('time_threshold'),
            **kwargs
        )
        from drl_negotiation.core.core import MySCML2020Agent
        self.parent: MySCML2020Agent = parent
        # kwargs['default_negotiator_type'] = default_negotiator_type
        # self.ufun = None

    def counter_all(
            self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """TODO: Calculate a response to all offers from all negotiators (negotiator ID is the key).

            Args:
                offers: Maps negotiator IDs to offers
                states: Maps negotiator IDs to offers AT the time the offers were made.

            Remarks:
                - The response type CANNOT be WAIT.
                - If the system determines that a loop is formed, the agent may receive this call for a subset of
                  negotiations not all of them.

        """
        if RANDOM:
            responses = {
                k: SAOResponse(random.choice(list(ResponseType)),
                               self.negotiators[k][0].ami.outcomes[
                                   random.randrange(0, len(self.negotiators[k][0].ami.outcomes))])
                for k in offers.keys()
            }
        else:
            responses = {}
            for nid in offers:
                negotiator = self.negotiators[nid]
                negotiation = [negotiation for negotiation in self.parent.running_negotiations
                               if negotiation.negotiator == self.negotiators[nid][0]][0]

                if negotiation.annotation["seller"] == self.parent.id:
                    index = self.parent.awi.my_consumers.index(negotiation.annotation["buyer"])
                    # TODO, convert action to legal outcome, the range of proposal
                    response_outcome = tuple(self.parent.action.m[index * 3:index * 3 + 3])
                else:
                    index = self.parent.awi.my_suppliers.index(negotiation.annotation["seller"])
                    # TODO, convert action to legal outcome
                    response_outcome = tuple(self.parent.action.b[index * 3:index * 3 + 3])

                response_type = ResponseType.ACCEPT_OFFER if offers[nid] == response_outcome \
                    else ResponseType.REJECT_OFFER

                logging.debug(f"offer is {offers[nid]} and response outcome is {response_outcome}")

                if offers[nid] == response_outcome:
                    print(f"Achieved, {offers[nid]} == {response_outcome}")

                responses[nid] = SAOResponse(
                    response_type,
                    None if response_type == ResponseType.ACCEPT_OFFER else response_outcome
                )
        #responses = super(MyDRLSCMLSAOSyncController, self).counter_all(offers, states)
        return responses


if __name__ == "__main__":
    pass
