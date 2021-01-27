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
        self.history_offers: Dict[str, "Outcome"] = {}
        self.history_running_negotiations = None
        # kwargs['default_negotiator_type'] = default_negotiator_type
        # self.ufun = None

    def respond(
        self, negotiator_id: str, state: MechanismState, offer: "Outcome"
    ) -> "ResponseType":
        # get the saved response to this negotiator if any
        response = self.responses.get(negotiator_id, None)
        if response is not None:
            # remove the response and return it
            del self.responses[negotiator_id]
            self.n_waits[negotiator_id] = 0
            return response

        # set the saved offer for this negotiator
        self.offers[negotiator_id] = offer
        self.offer_states[negotiator_id] = state
        n_negotiators = len(self.active_negotiators)
        # if we got all the offers or waited long enough, counter all the offers so-far
        if (
            len(self.offers) == n_negotiators
            or self.n_waits[negotiator_id] >= n_negotiators
        ):
            responses = self.counter_all(offers=self.offers, states=self.offer_states)
            for nid in responses.keys():
                # register the responses for next time for all other negotiators
                if nid != negotiator_id:
                    self.responses[nid] = responses[nid].response
                self.proposals[nid] = responses[nid].outcome
            self.offers = dict()
            self.offer_states = dict()
            self.n_waits[negotiator_id] = 0
            return responses[negotiator_id].response
        self.n_waits[negotiator_id] += 1
        return ResponseType.WAIT

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
            self.history_running_negotiations = self.parent.running_negotiations
            for nid in offers:
                self.history_offers[nid] = offers[nid]
                negotiator = self.negotiators[nid]
                negotiation = [negotiation for negotiation in self.parent.running_negotiations
                               if negotiation.negotiator == self.negotiators[nid][0]][0]

                if negotiation.annotation["seller"] == self.parent.id:
                    index = sorted(self.parent.awi.my_consumers).index(negotiation.annotation["buyer"])
                    # TODO, convert action to legal outcome, the range of proposal
                    response_outcome = tuple(self.parent.action.m[index * 3:index * 3 + 3])
                else:
                    index = sorted(self.parent.awi.my_suppliers).index(negotiation.annotation["seller"])
                    # TODO, convert action to legal outcome
                    response_outcome = tuple(self.parent.action.b[index * 3:index * 3 + 3])

                response_type = ResponseType.ACCEPT_OFFER if offers[nid] == response_outcome \
                    else ResponseType.REJECT_OFFER

                logging.debug(f"offer is {offers[nid]} and response outcome is {response_outcome}")

                if offers[nid] == response_outcome:
                    logging.info(f"Achieved, {offers[nid]} == {response_outcome}")

                responses[nid] = SAOResponse(
                    response_type,
                    None if response_type == ResponseType.ACCEPT_OFFER else response_outcome
                )
        #responses = super(MyDRLSCMLSAOSyncController, self).counter_all(offers, states)
        return responses


if __name__ == "__main__":
    pass
