from drl_negotiation.core.games._scml import MySCML2020Agent
from scml.scml2020 import (
    TradeDrivenProductionStrategy,
    PredictionBasedTradingStrategy,
)
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020 import (
    SupplyDrivenProductionStrategy,
    KeepOnlyGoodPrices,
    StepNegotiationManager,
)

from negmas import LinearUtilityFunction
from .mynegotiationmanager import MyNegotiationManager, MyConcurrentNegotiationManager


class MyComponentsBasedAgent(
    TradeDrivenProductionStrategy,
    MyNegotiationManager,
    PredictionBasedTradingStrategy,
    MySCML2020Agent
):
    """
        my components based agent
    """

    def create_ufun(self, is_seller: bool, issues=None, outcomes=None):
        """A utility function that penalizes high cost and late delivery 
                    for buying and and awards them for selling"""
        if is_seller:
            return LinearUtilityFunction((0, 0.25, 1))
        return LinearUtilityFunction((0, -0.5, -0.8))


class MyConcurrentBasedAgent(
    _NegotiationCallbacks,
    MyConcurrentNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    MySCML2020Agent
):
    """
        my concurrent based agent,
    """
    pass

class MyOpponentAgent(
    KeepOnlyGoodPrices,
    _NegotiationCallbacks,
    StepNegotiationManager,
    PredictionBasedTradingStrategy,
    SupplyDrivenProductionStrategy,
    MySCML2020Agent,
):
    def __init__(self, *arags, **kwargs):
        kwargs["adversary"] = True
        super().__init__(*arags, **kwargs)

