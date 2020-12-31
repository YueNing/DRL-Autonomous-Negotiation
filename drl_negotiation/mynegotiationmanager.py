'''
   negotiation manager, for scml agent
   Author: naodongbanana
   E-Mail: n1085633848@outlook.com
'''
from typing import List, Dict, Optional, Any
import functools
from scml import NegotiationManager
################ For Test #### will be removed #########
from scml.scml2020 import IndependentNegotiationsManager, StepNegotiationManager
########################################################
from negmas import AgentMechanismInterface, Negotiator, Issue, SAONegotiator
import numpy as np
from typing import Tuple, List
from scml.scml2020.components.negotiation import ControllerInfo
from .negotiator import MyDRLNegotiator
from .core import NegotiationRequestAction
from .controller import MyDRLSCMLSAOSyncController
from .hyperparameters import *
from drl_negotiation.utils import load_seller_neg_model
from drl_negotiation.utils import load_buyer_neg_model
from drl_negotiation.controller import MyDRLSCMLSAOSyncController

class MyNegotiationManager(IndependentNegotiationsManager):
    """
        my negotiation manager, strategy
    """

    def __init__(
            self,
            controller=MyDRLSCMLSAOSyncController,
            seller_model_path=NEG_SELL_PATH,
            buyer_model_path=NEG_BUY_PATH,
            load_model=LOAD_MODEL,
            train=TRAIN,
            * args,
            **kwargs
    ):
        super(MyNegotiationManager, self).__init__(*args, **kwargs)
        self.train = train
        self.seller_model = None
        self.buyer_model = None

        self.load_model = load_model
        if load_model:
            self.seller_model_path = seller_model_path
            self.buyer_model_path = buyer_model_path
        else:
            self.seller_model_path = None
            self.buyer_model_path = None

        if not train:
            self._load_model()
        # TODO: concurrent negotiation manager
        pass

    def _load_model(self, sell=True):
        if self.load_model:
            if self.seller_model_path is not None:
                self.seller_model = load_seller_neg_model(self.seller_model_path)

            if self.buyer_model_path is not None:
                self.buyer_model = load_buyer_neg_model(self.buyer_model_path)

    def respond_to_negotiation_request(
            self,
            initiator: str,
            issues: List[Issue],
            annotation: Dict[str, Any],
            mechanism: AgentMechanismInterface,
            ) -> Optional[Negotiator]:
        """
            IDEA 4.2: TODO: observation: financial report of initiator
                            action: ACCEPT or REJECT to negotiate
        """
        #import ipdb
        #ipdb.set_trace()
        
        #print(f'{self}: negotiation manager {self.action.m}, issues{issues}')
        #if self.action.m in ([NegotiationRequestAction.ACCEPT_REQUEST], 
        #                        [NegotiationRequestAction.DEFAULT_REQUEST]):
        return self.negotiator(annotation["seller"] == self.id, issues=issues)
        #return None

    def acceptable_unit_price(self, step: int, sell: bool) -> int:
        
        production_cost = np.max(self.awi.profile.costs[:, self.awi.my_input_product])
        if sell:
            return production_cost + self.input_cost[step]
        return self.output_price[step] - production_cost
    
    def target_quantity(self, step: int, sell:bool) -> int:
        """
            Idea 4.1. TODO: observation: negotiations, negotiation_requests
                         action: target quantity, discrete action_space
            return target quantity

        """
        if sell:
            needed, secured = self.outputs_needed, self.outputs_secured
        else:
            needed, secured = self.inputs_needed, self.inputs_secured

        return needed[step] - secured[step]

    def _start_negotiations(
            self,
            product: int,
            sell: bool,
            step: int,
            qvalues: Tuple[int, int],
            uvalues: Tuple[int, int],
            tvalues: Tuple[int, int],
            partners: List[str] = None,
            ) -> None:
        """
            IDEA 4.3: TODO: observation: market conditions, target_quantity, 
                                            acceptable_unit_price, negotiations, negotiation_requests,
                                            qvalues, uvalues, tvalues, step, sell
                            action: range of issues
        """
        # using model to predict the action
        _model = None
        if sell and self.seller_model is not None:
            _model = self.seller_model

        if not sell and self.buyer_model is not None:
            _model = self.buyer_model

        if _model is not None:
            # test period, get the action from model
            _obs = self._get_obs()
            _act = _model.action(_obs)

            self.action.s = np.zeros(DIM_S)

            if MANAGEABLE:
                if DISCRETE_ACTION_INPUT:
                    if _act[0] == 1: self.action.s[0] = -1.0
                    if _act[0] == 2: self.action.s[0] = +1.0
                    if _act[0] == 3: self.action.s[1] = -1.0
                    if _act[0] == 4: self.action.s[1] = +1.0
                else:
                    # one hot
                    if DISCRETE_ACTION_SPACE:
                        self.action.s[0] += _act[0][1] - _act[0][2]
                        self.action.s[1] += _act[0][3] - _act[0][4]
                    else:
                        self.action.s = _act[0]

                uvalues = tuple(np.array(uvalues) + np.array(self.action.s).astype("int32"))
        else:
            # training period, action has been set up in env
            if sell:
                if self.action.m is not None:
                    # set up observation
                    # self.state.o_role = sell
                    self.state.o_negotiation_step = self.awi.current_step
                    # for debug
                    self.state.o_step = step
                    self.state.o_is_sell = sell

                    self.state.o_q_values = qvalues
                    self.state.o_u_values = uvalues
                    self.state.o_t_values = tvalues

                    uvalues = tuple(np.array(uvalues) + (np.array(self.action.m)*self.action.m_vel).astype("int32"))
            else:
                # for buyer
                if self.action.b is not None:
                    uvalues = tuple(np.array(uvalues) + (np.array(self.action.b) * self.action.b_vel).astype("int32"))

        #import ipdb
        #ipdb.set_trace()
        #print(f"qvalues: {qvalues}, uvalues: {uvalues}, tvalues: {tvalues}")

        issues = [
                Issue(qvalues, name="quantity"),
                Issue(tvalues, name="time"),
                Issue(uvalues, name="uvalues")
                ]

        for partner in partners:
            self.awi.request_negotiation(
                    is_buy = not sell,
                    product = product,
                    quantity = qvalues,
                    unit_price = uvalues,
                    time = tvalues,
                    partner = partner,
                    negotiator = self.negotiator(sell, issues=issues)
                    )

class MyConcurrentNegotiationManager(StepNegotiationManager):

    def __init__(self):
        super().__init__()


    def add_controller(
        self,
        is_seller: bool,
        target,
        urange: Tuple[int, int],
        expected_quantity: int,
        step: int,
    ) -> MyDRLSCMLSAOSyncController:
        if is_seller and self.sellers[step].controller is not None:
            return self.sellers[step].controller
        if not is_seller and self.buyers[step].controller is not None:
            return self.buyers[step].controller
        controller = MyDRLSCMLSAOSyncController(
            is_seller=is_seller,
            target_quantity=target,
            negotiator_type=self.negotiator_type,
            negotiator_params=self.negotiator_params,
            step=step,
            urange=urange,
            product=self.awi.my_output_product
            if is_seller
            else self.awi.my_input_product,
            partners=self.awi.my_consumers if is_seller else self.awi.my_suppliers,
            horizon=self._horizon,
            negotiations_concluded_callback=functools.partial(
                self.__class__.all_negotiations_concluded, self
            ),
            parent_name=self.name,
            awi=self.awi,
        )
        if is_seller:
            assert self.sellers[step].controller is None
            self.sellers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        else:
            assert self.buyers[step].controller is None
            self.buyers[step] = ControllerInfo(
                controller,
                step,
                is_seller,
                self._trange(step, is_seller),
                target,
                expected_quantity,
                False,
            )
        return controller







