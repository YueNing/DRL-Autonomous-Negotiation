import inspect
from drl_negotiation.core.games._scml import MySCML2020Agent
from scml.scml2020 import (
    TradeDrivenProductionStrategy,
    PredictionBasedTradingStrategy,
)
from scml import QUANTITY, UNIT_PRICE
from scml.scml2020.agents.decentralizing import _NegotiationCallbacks
from scml.scml2020 import (
    SupplyDrivenProductionStrategy,
    KeepOnlyGoodPrices,
    StepNegotiationManager,
)

from negmas import LinearUtilityFunction, ResponseType
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

    def init(self):
        """
        Called once after the AWI is set.

        Remarks:
            - Use this for any proactive initialization code.
        """
        self.ufun.normalized = False

    def _random_offer(self, negotiator_id: str):
        return self.negotiators[negotiator_id][0].ami.random_outcomes(1)[0]

    def propose(self, negotiator_id: str, state: MechanismState, tag=False) -> "Outcome":
        if self.awi.agent.myoffer is None:
            return self.mypropose(negotiator_id, state)

        if tag:
            return self.mypropose(negotiator_id, state)

        return self.awi.agent.myoffer

    def mypropose(self, negotiator_id:str, state: MechanismState) -> "Outcome":
        action: int = self.policy(negotiator_id, state)
        # convert int action to propose
        if action == 0:
            return None

        action -= 1
        q = action // 101
        u = action % 101

        # print(f"World Step is: {self.awi.current_step}, propose is {(q, 0, u)}")
        return (q, self.awi.current_step, u)

        #return (q, 0, u)

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        self.awi.agent.myoffer = self.propose(negotiator_id, state, tag=True)
        if self.awi.agent.myoffer[QUANTITY] == offer[QUANTITY] and \
                self.awi.agent.myoffer[UNIT_PRICE] == offer[UNIT_PRICE]:
            return ResponseType.ACCEPT_OFFER
        return ResponseType.REJECT_OFFER
