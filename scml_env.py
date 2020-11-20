import gym
import numpy as np

from gym.utils import seeding
from gym.spaces import Discrete, Box
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Union, List, Tuple
from scml_game import (Game,
                                    DRLSCMLGame, 
                                    SCMLGame, 
                                    NegotiationGame, 
                                    DRLNegotiationGame,
                                    MyDRLNegotiationGame,
                                    MyDRLSCMLGame)

from negmas import AgentMechanismInterface

__all__ = [
    "BaseEnv",
    "DRLEnvMixIn",
    "RLEnvMixIn",
    "NEnv",
    "NegotiationEnv",
    "SCMLEnv",
    "DRLNegotiationEnv",
    "DRLSCMLEnv",
    "MyNegotiationEnv",
    "MySCMLEnv",

]

####################################################################################################
# For Env
#
#
####################################################################################################

class BaseEnv(ABC):
    '''
    Model the base environment class,
        1. basic running environment,
        2. drl running environment
        3. others environemnt using machine learning method except drl
    
    Attributs:
        name: The name of the running environment 
        game: The game(Negotiation Game or SCML Game)
        ami: Agent Mechanism Interface if it exists.

    Methods:
        step: 
        run:  
    '''

    def __init__(
        self,
        name: str = None,
        game: Optional[Game] = None,
    ):
        # super().__init__()
        self._name = name
        self._game = game

    def __str__(self):
        return f'The name of Env is {self.name}'
    
    @property
    def name(self):
        return self._name 

    @property
    def game(self):
        return self._game
    
    def step(self, action=None):
        '''
        Run the env step by step
        '''
        self.game.step()

    def run(self):
        '''
        Run directly 
        '''
        self.game.run()
    
    def seed(self, seed=None):
        # get the np random RandomState and strong seed number
        self.np_random, seed = seeding.np_random(seed)
        # set the np random RandomState in game
        self.game.seed(self.np_random)

        return [seed]


class RLEnvMixIn:
    '''
    Some mixin function of reinforcement learning
    '''
    def set_observation_space(self, observation_space: Union[List, Tuple, Box] = None):
        '''
        Check the inputed observation space valid or invalid
        '''

        if type(observation_space) == Box:
            self._observation_space = observation_space
        elif type(observation_space) == list or type(observation_space) == tuple:
            self._observation_space = Box(
                    low=np.array(observation_space[0]),
                    high=np.array(observation_space[1]),
                    dtype=np.int
                )

        return self._observation_space

    def set_action_space(self, action_space: Union[int, gym.spaces.Box, gym.spaces.Discrete, List, Tuple] = None):
        '''
        As Same as the function set_observation_space
        '''

        if type(action_space) == int:
            self._action_space = Discrete(action_space)
        if type(action_space) == Discrete or type(action_space) == Box:
            self._action_space = action_space

        if type(action_space) == List or type(action_space) == Tuple:
            self._action_space = Box(
                low=action_space[0],
                high=action_space[1],
                dtype=np.int
            )

        return self._action_space
    
    @property
    def get_action_space(self):
        """Return the action space"""
        return self._action_space
    
    @property
    def get_observation_space(self):
        """Return the observation space"""
        return self._observation_space

    @staticmethod
    def check(observation_space, action_space, game_type="DRLNegotiation", strategy="ac_s"):
        """

        Args:
            observation_space: observation space
            action_space: action space
            game_type: DRLNegotiation, DRLSCML
            strategy: if game_type is DRLNegotiation, strategy is ac_s,

        Returns:
            bool
        """
        if game_type == "DRLNegotiation":
            assert isinstance(observation_space, gym.spaces.Space), "Error Observation Space!"

            if strategy == "ac_s":
                assert isinstance(action_space, gym.spaces.Discrete), "Error Action Space!"
            if strategy == "of_s":
                assert isinstance(action_space, gym.spaces.Box), "Error Action Space, must be gym.spaces.B"

        return True

class DRLEnvMixIn:
    '''
    TODO: Some Mixin function of deep reinforcement learning
    '''
    def network(self):
        pass

class NEnv(RLEnvMixIn, BaseEnv, gym.Env, ABC):
    """
        All of the deep reinforcement learning environment class inherit from this class

        Abstract method:

            get_obs, how to design the observation space and based on it to get observatin is important for reinforcement learning
            render, inherit from gym.Env, different game type will render different information, so set here abstract method here also.
            close, the same as the method render
    """

    def __init__(
        self,
        name: str = None,
        game: Optional[Game] = None,
    ):
        """

        Args:
            name: the name of environment
            game: negotiation game or scml game
        """
        super(NEnv, self).__init__(
            name=name,
            game=game,
        )
        # super().__init__()
    
        self.n_step = None
        self.time = None

    def init(self):

        self.n_step = 0
        self.time = 0

        print(f"Initial the environment {self.name}, game is {self.game.name}, ")

    @property
    def action_space(self):
        return self.get_action_space
    
    @property
    def observation_space(self):
        return self.get_observation_space

    @abstractmethod
    def get_obs(self):
        raise NotImplementedError("Error: function get_obs in DRLEnvMixIn has not been implemented!")
        
    def _get_obs(self) -> "gym.spaces.Box.sample()":
        return self.get_obs()

    def step(self, action: "gym.spaces.Discrete.sample" = None):
        done = False
        self.n_step += 1
        self.time = getattr(self.game.ami.state, 'relative_time')

        # Run one step forward, and return the reward from game
        reward = self.game.step(action=action)

        obs = self._get_obs()

        # meet the condition: done = True
        if self.n_step >= getattr(self.game.ami, 'n_steps') \
            or self.time >= float(getattr(self.game.ami.state, 'relative_time')):
            done = True
        
        if not self.game.get_life():
            done = True
        
        # infos
        info = {
            'state': self.game.get_state()
        }
        
        return obs, reward, done, info
    
    @abstractmethod
    def render(self, mode='human', close=False):
        raise NotImplementedError('')

    @abstractmethod
    def close(self):
        raise NotImplementedError('')

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation,
        initial observation obtains negotiators' intial observation and others information 
        relates to the definition of observation space. 
        """
        # env reset
        self.init()

        # game reset
        self.game.init_game()

        # get initial observation
        init_obs = self._get_obs()

        return init_obs


####################################################################################################
# For Negotiation
#
#
####################################################################################################

class NegotiationEnv(NEnv):
    '''
    RL base class, default settings, ideas comes from ANegma,
    can passed through drl method such as, dql, ppo1,
    but just under the default settings defined in stable_baselines,
    '''
    def __init__(
        self,
        name: str = "NegotiationEnv",
        game: Optional[NegotiationGame] = None,
        strategy: Union["ac_s", "of_s", "hybrid"] = "ac_s",
        observation_space: Union[List[List[int]], List[List[float]], None] = None,
        action_space: Optional[int] = None,
    ):
        """

        Args:
            name: The name of negotiation environment
            game: the name of negotiation game
            strategy:
                ac_s: acceptance strategy, when the learned strategy is acceptance strategy
                of_s: offer/bidding strategy, when the learned strategy is offer/bidding strategy
                hybrid: both acceptance and offer/bidding strategy, when both acceptance strategy and offer/bidding are learned,
                in other words, both discrete action space and continuous action space are existed in the learned strategy.
            observation_space: observation space used by model
            action_space: action space used by model
        """
        if game is None:
            game = NegotiationGame(
                name="negotiation_game",
                env=self
            )

        super().__init__(
            name=name,
            game=game,
        )
        self._strategy = strategy

        # TODO: set the observation space, for reinforcement learning
        # Default ANEGMA, single issue
        # state: ("Xbest":, "Tleft":, "IP_my", "RP_my")
        # observ_space = spaces.Box(low=np.array([300, 0, 300, 500]), high=np.array([550, 1, 350, 550]), dtype=np.int)

        if observation_space is None:
            observation_space = [[300, 0, 300, 500], [550, 210, 350, 550]]

        self.set_observation_space(observation_space=observation_space)

        # TODO: set the action space ....
        if action_space is None:
            if self.strategy == "ac_s":
                # Default 5 actions: REJECT, ACCEPT, END, NO_RESPONSE, WAIT
                action_space = gym.spaces.Discrete(5)
            elif self.strategy == "of_s":
                action_space = gym.spaces.Box(
                    low=self.game.format_issues[0],
                    high=self.game.format_issues[1],
                    dtype=np.int
                )
            elif self.strategy == "hybrid":
                action_space = self.hybrid_action_space()

        self.set_action_space(action_space=action_space)
        # import pdb;pdb.set_trace()

        self.check(self.observation_space, self.action_space, self.game.game_type, self.strategy)

    def hybrid_action_space(self):
        raise NotImplementedError("The hybrid action_space is not Implemented in this Class, "
                                  "please firstly implements method hybrid_action_space!"
                                  "or set the strategy as ac_s or of_s!")

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def get_obs(self):
        """
        Condition 1: For negotiator perspective,
        Returns:
            obs, [quantity, ]
        """
        obs = self.game.get_observation()
        return obs

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

class DRLNegotiationEnv(DRLEnvMixIn, NegotiationEnv):
    '''
    DRL, common setting based on issues, deep reinfoment learning,
    Mainly responsible for hybrid discrete-continuous reinforcement learning

    Train a negotiator that based on the settings extracted from scml or defined by user.
    issues will be defined in a game class, used here in order to generate the observation space and
    action space that later will be used by train model.
    '''
    def __init__(
        self,
        name:str = 'DRLNegotiationEnv',
        game: Optional[DRLNegotiationGame] = None,
        strategy: Union["ac_s", "of_s", "hybrid"] = "ac_s",
        observation_space: Union[List[List[int]], List[List[float]], None] = None,
        action_space: Optional[int] = None,
    ):
        # My Negotiation observation space and action space
        # issues = [[0, 0, 0], [100, 10, 100]], observation_space = [issues, current_time]
        # multi issues, [quantity, time, unit_price] => [[0, 0, 0], [1, 1, 1]], normalization implemented by drl algorithm!

        if game is None:
            game = DRLNegotiationGame(
                name=name,
                env=self,
            )

        # no user defined observation space and action space, use the default setting,
        # get the issues from simulator and combine the current time as the observation space
        # overwrite the super default observation_space and action space when the inputed observation_space or action_space is None.
        if observation_space is None:
            observation_space = [
                game.format_issues[0] + [0],
                game.format_issues[1] + [game.n_steps],
            ]

        if action_space is None:
            action_space = 3

        super().__init__(
            name=name,
            game=game,
            strategy=strategy,
            observation_space=observation_space,
            action_space=action_space,
        )

    def hybrid_action_space(self) -> gym.spaces.Space:
        """
        TODO: for hybrid action space
        Returns:
            gym.spaces.Space
        """
        pass



class MyNegotiationEnv(DRLNegotiationEnv):

    def __init__(
        self,
        name: str = "my_negotiation_env",
        game: Optional[DRLNegotiationGame] = None,

    ):
        # init my drl negotiation game
        if game is None:
            game = MyDRLNegotiationGame(
                name='my_drl_negotiation_game',
                env=self
            )

        super().__init__(
            name=name,
            game=game,
        )


####################################################################################################
# For SCML
#
#
####################################################################################################

class SCMLEnv(NEnv):
    
    def __init__(
        self,
        name: str = "SCMLEnv",
        game: Optional[SCMLGame] = None,
        mechanism_state: "negmas.MechanismState" = None,
    ):
        super().__init__(
            name=name,
            game=game,
        )
        self._mechanism_state = mechanism_state
    
    def mechanism_state(self):
        return self._mechanism_state

class DRLSCMLEnv(SCMLEnv):

    def __init__(
        self,
        name: str="DRLSCMLEnv",
        game: Optional[DRLSCMLGame] = None,
        observation_space: Optional[gym.spaces.Box] = None,
        action_space: Optional[gym.spaces.Discrete] = None,
        mechanism_state: "negmas.MechanismState" = None,
    ):
        super().__init__(
            name=name,
            game=game,
            mechanism_state=mechanism_state
        )
        self.set_observation_space(observation_space=observation_space)
        self.set_action_space(action_space=action_space)
    
    def get_obs(self):
        pass
    

class MySCMLEnv(DRLSCMLEnv):

    def __init__(
        self,
        name: str = "MySCMLEnv",
        game: Optional[Game] = MyDRLSCMLGame,
    ):
        super().__init__(
            name=name,
            game=game,
        )
    
    def __str__(self):
        return f"The name of MySCMLEnv is {self.name}"