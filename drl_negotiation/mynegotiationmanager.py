'''
   negotiation manager, for scml agent
   Author: naodongbanana
   E-Mail: n1085633848@outlook.com
'''
from typing import List, Dict, Optional, Any

from scml import NegotiationManager
################ For Test #### will be removed #########
from scml.scml2020 import IndependentNegotiationsManager
########################################################
from negmas import AgentMechanismInterface, Negotiator, Issue, SAONegotiator
import numpy as np
from typing import Tuple, List
from .negotiator import MyDRLNegotiator
from .core import NegotiationRequestAction
from .controller import MyDRLSCMLSAOSyncController
from .hyperparameters import *
from drl_negotiation.utils import load_seller_neg_model
from drl_negotiation.utils import load_buyer_neg_model
from drl_negotiation.utils import get_trainers
from drl_negotiation.utils import reverse_normalize
from drl_negotiation.a2c.policy import create_actor
import drl_negotiation.utils as U
from gym import spaces
import tensorflow as tf

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
        self.seller_model = '/tmp/policy/'
        self.buyer_model = '/tmp/policy/'

        self.load_model = load_model
        self.train = train

        if load_model:
            self.seller_model_path = seller_model_path
            self.buyer_model_path = buyer_model_path
        else:
            self.seller_model_path = None
            self.buyer_model_path = None
        #
        # if not train:
        #     self._setup_model()

    def _setup_model(self):
        """
        get the buyer and seller trainer/model,
        create the policy network and load the saved parameters
        Returns:

        """
        scopes = [self.name.replace("@", '-') + "_seller", self.name.replace('@', '-') + "_buyer"]
        obs_ph = []

        # observation space
        observation_space = []
        obs_dim = [len(self._get_obs(seller=True)), len(self._get_obs(seller=False))]

        observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim[0],), dtype=np.float32))
        observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim[1],), dtype=np.float32))

        obs_shape = [observation_space[i].shape for i in range(len(observation_space))]

        for i in range(len(obs_shape)):
            obs_ph.append(U.BatchInput(obs_shape[i], name="observation" + str(i)).get())

        make_obs_ph = obs_ph

        # action space
        if DISCRETE_ACTION_SPACE:
            act_space = [spaces.Discrete(DIM_M*2 + 1), spaces.Discrete(DIM_B*2 + 1)]
        else:
            act_space = [spaces.Box(low=-self.m_range, high=+self.m_range, shape=(DIM_M, ), dtype=np.float32),
                        spaces.Box(low=-self.m_range, high=+self.m_range, shape=(DIM_B, ), dtype=np.float32)]

        self.models = []

        for index in range(len(scopes)):
            self.models.append(
                (create_actor(
                    make_obs_ph=make_obs_ph[index],
                    act_space=act_space[index],
                    scope=scopes[index]
                ), scopes[index]))

        self.scopes = scopes

    def _load_state(self, model):
        """
        TODO
        restore the model
        Args:
            sell:

        Returns:

        """
        pass

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

    def _urange(self, step, is_seller, time_range):
        """
        TODO:
        use action from maddpg to set the urange,
        Args:
            step:
            is_seller:
            time_range:

        Returns:

        """
        if is_seller:
            cprice = self.awi.catalog_prices[self.awi.my_output_product]
            urange = (cprice, 2 * cprice)
        else:
            cprice = self.awi.catalog_prices[self.awi.my_input_product]
            urange = (1, cprice)

        if DISCRETE_ACTION_SPACE:
            return urange
        else:
            _model = None
            _obs = None
            if is_seller and self.seller_model is not None:
                _model = 'seller'

            if not is_seller and self.buyer_model is not None:
                _model = "buyer"

            if _model is not None:
                self._load_model()
                _obs = self._get_obs(seller=is_seller)
                tag = 0 if _model == 'seller' else 1
                # _act =[-1, 1]
                _act = self.trainers[tag].action(_obs)
                if tag:
                    mul_1 = (cprice, 3/2 * cprice)
                    # (3/2 cprice, 2 cprice)
                    mul_2 = (3/2 * cprice, 2 * cprice)
                else:

                    mul_1 = (1, 1/2 * cprice)
                    # (1/2 cprice, cprice)
                    mul_2 = (1/2 * cprice, cprice)

                urange = reverse_normalize(tuple(_act), (mul_1, mul_2))
                return urange
            else:
                # for training
                if is_seller:
                    if self.action.m is not None:
                        urange = self.action.m
                else:
                    if self.action.b is not None:
                        urange = self.action.b

                return urange


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
        if DISCRETE_ACTION_SPACE:
            _model = None
            if not self.train and RUNNING_IN_SCML2020World:
                if sell:
                    _model = self.models[0]

                if not sell:
                    _model = self.models[1]

            if _model is not None:
                #TODO: test period, get the action from model
                with U.single_threaded_session():
                    U.initialize()
                    self._load_state(_model)

                    _obs = self._get_obs()
                    _act = _model[0](_obs[None])[0]

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

                        if self.action.m is not None:
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












