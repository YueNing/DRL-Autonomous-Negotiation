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
                 parent:"PredictionBasedTradingStrategy",
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
        responses = {
            k: SAOResponse(random.choice(list(ResponseType)),
                           self.negotiators[k][0].ami.outcomes[random.randrange(0, len(self.negotiators[k][0].ami.outcomes))])
            for k in offers.keys()
        }
        return responses
        # return super(MyDRLSCMLSAOSyncController, self).counter_all(offers, states)

if __name__ == "__main__":
    pass
