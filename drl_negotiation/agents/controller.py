'''
DRL Controller
'''
############################
# Packages used for Test
from negmas import LinearUtilityFunction
from scml import (IndependentNegotiationsManager, 
                    PredictionBasedTradingStrategy, 
                        DemandDrivenProductionStrategy, 
                            RandomAgent,
                            DecentralizingAgent,
                                SCML2020Agent)
import matplotlib.pyplot as plt
############################
from typing import Union, Optional, Dict, Tuple
from negmas import (SAOSyncController,
                    MechanismState, 
                    SAONegotiator,
                    AspirationNegotiator,
                    Outcome,
                    SAOResponse,
                    SAOState)

from scml.scml2020.services import StepController, SyncController

from scml import SCML2020World, NegotiationManager



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
                 parent:str = "PredictionBasedTradingStrategy",
                 is_seller: bool = None,
                 default_negotiator_type ="MyDRLNegotiator",
                 *args,
                 **kwargs
        ):
        super().__init__(
            is_seller=is_seller,
            parent=parent,
            price_weight = kwargs.pop('price_weight'),
            utility_threshold = kwargs.pop('utility_threshold'),
            time_threshold = kwargs.pop('time_threshold'),
            **kwargs
        )
        # kwargs['default_negotiator_type'] = default_negotiator_type
        # self.ufun = None

    def best_proposal(self, nid: str) -> Tuple[Optional[Outcome], float]:
        # TODO: proposal
        return  super().best_proposal(nid=nid)

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
        pass

if __name__ == "__main__":
    pass
