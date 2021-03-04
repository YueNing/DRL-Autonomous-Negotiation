"""
DRL Controller
"""
############################
# Packages used for Test
############################
import logging
import copy
import random
from typing import Dict
from negmas import (Outcome,
                    SAOResponse,
                    SAOState,
                    MechanismState,
                    )

from scml.scml2020.services import SyncController
import numpy as np
from scml.scml2020.common import UNIT_PRICE
from negmas import ResponseType
from drl_negotiation.core.config.hyperparameters import RANDOM


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
        self.parent = parent
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

        # saved for observation
        self.history_running_negotiations = self.parent.running_negotiations
        self.history_offers = copy.deepcopy(self.offers)

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
            for nid in offers:
                # self.history_offers[nid] = offers[nid]
                negotiator = self.negotiators[nid]
                try:
                    negotiation = [negotiation for negotiation in self.parent.running_negotiations
                                   if negotiation.negotiator == self.negotiators[nid][0]][0]
                except Exception as e:
                    print("error")

                if negotiation.annotation["seller"] == self.parent.id:
                    index = sorted(self.parent.awi.my_consumers).index(negotiation.annotation["buyer"])
                    # TODO, convert action to legal outcome, the range of proposal
                    response_outcome = tuple(np.array(self.parent.action.m[index * 3:index * 3 + 3], dtype=int))
                else:
                    index = sorted(self.parent.awi.my_suppliers).index(negotiation.annotation["seller"])
                    # TODO, convert action to legal outcome
                    response_outcome = tuple(np.array(self.parent.action.b[index * 3:index * 3 + 3], dtype=int))


                util = self.utility(offers[nid], self.negotiators[nid][0].ami.issues[UNIT_PRICE].max_value)
                response_outcome_utility = self.utility(response_outcome, self.negotiators[nid][0].ami.issues[UNIT_PRICE].max_value)

                #if util >= 0.99 * response_outcome_utility and util > 0:
                if response_outcome in self.negotiators[nid][0].ami.outcomes:
                    if offers[nid] == response_outcome:
                        response_type = ResponseType.ACCEPT_OFFER
                        logging.debug(f"ACCEPT: offer is {offers[nid]} and response outcome is {response_outcome}")
                    else:
                        response_type = ResponseType.REJECT_OFFER
                        logging.debug(f"REJECT: offer is {offers[nid]} and response outcome is {response_outcome}")
                    self.parent.reward.append(1)
                else:
                    response_type = ResponseType.ACCEPT_OFFER
                    logging.debug(f"WAIT: offer is {offers[nid]} and response outcome is {response_outcome}, issues are "
                                  f"{self.negotiators[nid][0].ami.issues}")
                    self.parent.reward.append(-1)

                if response_type == ResponseType.ACCEPT_OFFER:
                    logging.debug(f"Achieved, {offers[nid]} == {response_outcome}")

                responses[nid] = SAOResponse(
                    response_type,
                    None if response_type == ResponseType.ACCEPT_OFFER
                            or response_type == ResponseType.WAIT
                    else response_outcome
                )

        #responses = super(MyDRLSCMLSAOSyncController, self).counter_all(offers, states)
        return responses


if __name__ == "__main__":
    pass
