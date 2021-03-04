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
from drl_negotiation.core.games._scml_oneshot import MyOneShotAgent
from negmas import MechanismState

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


class MyOneShotBasedAgent(MyOneShotAgent):

    def _random_offer(self, negotiator_id: str):
        return self.negotiators[negotiator_id][0].ami.random_outcomes(1)[0]

    def propose(self, negotiator_id: str, state: MechanismState) -> "Outcome":
        from scml.oneshot import QUANTITY, UNIT_PRICE
        action: int = self.policy(negotiator_id, state)
        # convert int action to propose
        q = action // 100 + 1
        u = action % 100 + 1
        return (q, 0, u)
