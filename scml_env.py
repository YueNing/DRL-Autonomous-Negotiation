import gym
from gym.utils import seeding

from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, Union, List
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
    def set_observation_space(self, observation_space: Optional[List] = None):
        '''
        Check the inputed observation space valid or invalid
        '''
        from gym.spaces import Box
        import numpy as np 

        self._observation_space = Box(
                low=np.array(observation_space[0]),
                high=np.array(observation_space[1]),
                dtype=np.int
            )

        return self._observation_space
    
    def set_action_space(self, action_space: Optional[int] = None):
        '''
        As Same as the function set_observation_space
        '''
        from gym.spaces import Discrete

        self._action_space = Discrete(action_space)
        return self._action_space
    
    @property
    def get_action_space(self):
        """Return the action space"""
        return self._action_space
    
    @property
    def get_observation_space(self):
        """Return the observation space"""
        return self._observation_space

class DRLEnvMixIn:
    '''
    TODO: Some Mixin function of deep reinforcement learning
    '''
    def network(self):
        pass

class NEnv(RLEnvMixIn, BaseEnv, gym.Env, ABC):
    '''
        All of the deep reinforcement learning environment class inherit from this class
        Abstract method
            observation_space: set the attribut, _observation_space
        
    '''

    def __init__(
        self,
        name: str = None,
        game: Optional[Game] = None,
    ):
        super(NEnv, self).__init__(
            name=name,
            game=game,
        )
        # super().__init__()
    
        self.n_step = None
        self.time = None
    
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
        self.n_step = 0
        self.time = 0

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
    RL base class, default settings, ideas comes from ANegma
    '''
    def __init__(
        self,
        name: str = "NegotiationEnv",
        game: Optional[NegotiationGame] = None,
        observation_space: Union[List[List[int]], List[List[float]], None] = None,
        action_space: Optional[int] = None,
    ):
        super().__init__(
            name=name,
            game=game,
        )

        from gym.spaces import Box, Discrete
        import numpy as np 

        # TODO: set the observation space, for reinforcement learning
        # Default ANEGMA, single issue
        # state: ("Xbest":, "Tleft":, "IP_my", "RP_my")
        # observ_space = spaces.Box(low=np.array([300, 0, 300, 500]), high=np.array([550, 1, 350, 550]), dtype=np.int)

        if observation_space is None:
            observation_space = [[300, 0, 300, 500], [550, 210, 350, 550]]
        
        self.set_observation_space(observation_space=observation_space)

        # TODO: set the action space ....
        # Default 5 actions: REJECT, ACCEPT, END, NO_RESPONSE, WAIT
        if action_space is None:
            action_space = 5
        
        self.set_action_space(action_space=action_space)
        # import pdb;pdb.set_trace()
    

    def get_obs(self):
        """
        Condition 1: For negotiator perspective,
        Returns:
            obs
        """
        obs = self.game.get_observation()
        return obs

    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

class DRLNegotiationEnv(NegotiationEnv):
    '''
    DRL, common setting based on issues, 
    
    Train a negotiator that will based on the settings extracted from scml or defined by user.
    issues will be defined in a game class, used here to generate the observation space and 
    action space that later will be used by train model.
    '''
    def __init__(
        self,
        name:str = 'DRLNegotiationEnv',
        game: Optional[DRLNegotiationGame] = None,
        observation_space: Union[List[List[int]], List[List[float]], None] = None,
        action_space: Optional[int] = None,
    ):
        # My Negotiation observation space and action space
        # issues = [[0, 0, 0], [100, 10, 100]], observation_space = [issues, current_time]
        # multi issues, [quantity, time, unit_price] => [[0, 0, 0], [1, 1, 1]], normalization implemented by drl algorithm!

        if game is None:
            game = DRLNegotiationGame(
                name=name
            )
        
        super().__init__(
            name=name,
            game=game,
            observation_space=observation_space,
            action_space=action_space,
        )


        # no user defined observation space and action space, use the default setting, 
        # get the issues from simulator and combine the current time as the observation space
        # overwrite the super default observation_space and action space when the inputed observation_space or action_space is None.
        
        if observation_space is None:
            observation_space = [
                                    self.game.format_issues[0] + [0], 
                                    self.game.format_issues[1] + [self.game.n_steps],
                                ]
            self.set_observation_space(observation_space=observation_space)
        
        if action_space is None:
            action_space = 3
            self.set_action_space(action_space=action_space)



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