from .core import MySCML2020Agent
from scml.scml2020 import (
        TradeDrivenProductionStrategy,
        PredictionBasedTradingStrategy,
        )
from negmas import LinearUtilityFunction
from .mynegotiationmanager import MyNegotiationManager

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


