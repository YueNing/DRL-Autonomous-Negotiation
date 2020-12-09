'''
DRL Controller
'''
############################
# Packages used for Test
from collections import defaultdict
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
from negmas import (SAOController, 
                    MechanismState, 
                    SAONegotiator,
                    AspirationNegotiator,
                    get_class)

from scml import SCML2020World, NegotiationManager

class DRLSCMLController(SAOController):
    """
    A Controller that used by Deep Reinforcement learning method

    Args: 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def propose(self, negotiator_id: str, state: MechanismState):
        pass
    
    def response(self, negotiator_id:str, state:MechanismState, offer: "Outcome") -> "ResponseType":
        pass

class DRLNegotiationManager(NegotiationManager):
    """
    Negotiation Manager that manages dlr negotiatiors that negotiator uses dlr method,
    TODO: negotiator_type will be replaced by my dlr sao negotiator
    """

    def __init__(self,
                *args,
                negotiator_type: Union[SAONegotiator, str] = AspirationNegotiator,
                negotiator_params: Optional[Dict[str, any]] = None,
                **kwargs
                ):
        super().__init__(*args, **kwargs)
        
        self.negotiator_type = get_class(negotiator_type)
        self.negotiator_params = (
            negotiator_params if negotiator_params is not None else dict()
        )

        self.buyers = self.sellers = None
    
    def init(self):
        super().init()
    
    def respond_to_negotiation_request(self):
        pass

class MyAgent(IndependentNegotiationsManager, PredictionBasedTradingStrategy, DemandDrivenProductionStrategy, SCML2020Agent):
    '''
    My Agent used drl,
    '''
    def target_quantity(self, step: int, sell: bool) -> int:
        return self.awi.n_lines // 2
    
    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        return self.awi.catalog_prices[self.awi.my_output_product] if sell else self.awi.catalog_prices[self.awi.my_input_product]

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        
        if is_seller:
            return LinearUtilityFunction((0, 0.25, 1))
        return LinearUtilityFunction((0, -0.5, -0.8))

def show_agent_scores(world):
    scores = defaultdict(list)
    for aid, score in world.scores().items():
        scores[world.agents[aid].__class__.__name__.split(".")[-1]].append(score)
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    plt.bar(list(scores.keys()), list(scores.values()), width=0.2)
    plt.show()

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
