'''
   negotiation manager, for scml agent
   Author: naodongbanana
   E-Mail: n1085633848@outlook.com
'''
import os, sys
import random
from typing import List, Dict, Optional, Any
import functools
from scml import NegotiationManager
################ For Test #### will be removed #########
from scml.scml2020 import IndependentNegotiationsManager, StepNegotiationManager
########################################################
from negmas import AgentMechanismInterface, Negotiator, Issue, SAONegotiator
import numpy as np
import logging
from typing import Tuple, List
from scml.scml2020.components.negotiation import ControllerInfo
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
        self.model_path = None
        self.load_model = load_model
        self.train = train
        self.saver = {}
        self.already_loaded = []
        self.initalize = []

        if load_model:
            self.seller_model_path = seller_model_path
            self.buyer_model_path = buyer_model_path
        else:
            self.seller_model_path = None
            self.buyer_model_path = None
        #
        # if not train:
        #     self._setup_model()

    @staticmethod
    def _search_policies(dirs):
        policies = []
        for _ in dirs:
            policies.append(U.traversal_dir_first_dir(_))
        return policies

    def _setup_model(self):
        """
        get the buyer and seller trainer/model,
        create the policy network and load the saved parameters
        Returns:

        """
        print(f"setup model called!{self}")
        dirs = ['/tmp/policy/', '/tmp/policy3/', '/tmp/policy4/']
        policies = self._search_policies(dirs)

        scope_prefix = self.name.replace("@", '-')
        scopes = [scope_prefix + "_seller", scope_prefix + "_buyer"]
        _tmp_index = []
        for index, policy in enumerate(policies):
            if dirs[index]+scopes[0] in policy and dirs[index] + scopes[1] in policy:
                _tmp_index.append(index)
        if not _tmp_index:
            logging.info(f"Do not load trained model for {self}, use default logic!")
            print(f"Do not load trained model for {self}, use default logic!")
            return
        else:
            self.model_path = dirs[random.choice(_tmp_index)]

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
        restore the model
        Args:
            sell:

        Returns:

        """
        logging.info("loading model")
        if os.path.exists(self.model_path+model[1]):
            if self.name+model[1] not in self.saver:
                var = tf.global_variables()
                var_flow_restore = [val for val in var if model[1] in val.name]
                #self.saver[self.name] = tf.train.import_meta_graph(self.model_path+model[1]+'/'+MODEL_NAME+'.meta')
                self.saver[self.name] = tf.train.Saver(var_flow_restore)

            # if self.name+model[1] not in self.already_loaded:
            U.load_state(tf.train.latest_checkpoint(self.model_path+model[1]), saver=self.saver[self.name])
            self.already_loaded.append(self.name+model[1])

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
            if is_seller and self.model_path is not None:
                _model = 'seller'

            if not is_seller and self.model_path is not None:
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
                if sell and self.model_path is not None:
                    _model = self.models[0]

                if not sell and self.model_path is not None:
                    _model = self.models[1]

            if _model is not None:
                #TODO: test period, get the action from model
                with U.single_threaded_session():
                    if self.name+_model[1] not in self.already_loaded:
                        if self.name+_model[1] not in self.initalize:
                            U.initialize()
                            self.initalize.append(self.name+_model[1])
                        self._load_state(_model)

                    _obs = self._get_obs()
                    try:
                        _act = _model[0](_obs[None])
                    except Exception as e:
                        self._load_state(_model)
                        _act = _model[0](_obs[None])

                    if MANAGEABLE:
                        self.action.s = np.zeros(DIM_S)
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







