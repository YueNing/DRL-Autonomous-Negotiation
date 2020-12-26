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
from typing import Union, Optional, Dict
from negmas import (SAOSyncController,
                    MechanismState, 
                    SAONegotiator,
                    AspirationNegotiator,
                    SAOResponse,
                    SAOState)

from scml import SCML2020World, NegotiationManager



##########################################################################################################
# Controller for SCML, Used for training concurrent negotiation with DRL
# Author naodongbanana
# Datum 25.12.2020
##########################################################################################################
class MyDRLSCMLSAOSyncController(SAOSyncController):
    """
    TODO:
    A Controller that used by Deep Reinforcement learning method, can manage multiple negotiators synchronously

    Args:
    """

    def __init__(self,
                 parent:str = "PredictionBasedTradingStrategy",
                 is_seller: bool = None,
                 default_negotiator_type ="MyDRLNegotiator",
                 *args,
                 **kwargs
        ):
        kwargs['default_negotiator_type'] = default_negotiator_type
        super().__init__(*args, **kwargs)
        self._is_seller = is_seller
        self.__parent = parent
        self.ufun = None


    def counter_all(
        self, offers: Dict[str, "Outcome"], states: Dict[str, SAOState]
    ) -> Dict[str, SAOResponse]:
        """Calculate a response to all offers from all negotiators (negotiator ID is the key).

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
    ComparisonAgent = DecentralizingAgent
    world = SCML2020World(
        **SCML2020World.generate([ComparisonAgent, MyAgent, RandomAgent], n_steps=10),
        construct_graphs=True,
    )
    world.run()
    world.draw(steps=(0, world.n_steps), together=False, ncols=2, figsize=(20, 20))
    plt.show()

    show_agent_scores(world)
