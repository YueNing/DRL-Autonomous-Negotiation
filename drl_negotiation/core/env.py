import gym
import numpy as np
from gym.utils import seeding
from gym.spaces import Discrete, Box
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
from .game import (Game,
                   NegotiationGame,
                   DRLNegotiationGame,
                   MyDRLNegotiationGame,
                   )

__all__ = [
    "BaseEnv",
    "DRLEnvMixIn",
    "RLEnvMixIn",
    "NEnv",
    "NegotiationEnv",
    "SCMLEnv",
    "DRLNegotiationEnv",
    "MyNegotiationEnv",
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
        self._game.set_env(self)

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

        if type(action_space) == list or type(action_space) == tuple:
            self._action_space = Box(
                low=np.array(action_space[0]),
                high=np.array(action_space[1]),
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
    Some Mixin function of deep reinforcement learning
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
        self.init()

    def init(self):

        self.n_step = 0
        self.time = 0

        # print(f"Initial the environment {self.name}, game is {self.game.name}, ")

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
        obs: List = self.get_obs()
        return np.array(obs)

    def step(self, action: "gym.spaces.Discrete.sample" = None):
        done = False
        self.n_step += 1
        self.time = getattr(self.game.ami.state, 'relative_time')

        # Run one step forward, and return the reward from game
        reward = self.game.step(action=action)

        obs = self._get_obs()

        # meet the condition: done = True
        if self.n_step >= self.game.ami.n_steps \
            or self.time >= self.game.ami.time_limit:
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

    def reset(self) -> object:
        """
        Resets the environment to an initial state and returns an initial observation,
        initial observation obtains negotiators' intial observation and others information 
        relates to the definition of observation space. 

        Returns:
            object: the initial observation
        """
        # env reset
        self.init()

        # game reset
        self.game.reset()

        # get initial observation
        init_obs: np.array = self._get_obs()

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

        # set the observation space, for reinforcement learning
        # Default ANEGMA, single issue
        # state: ("Xbest":, "Tleft":, "IP_my", "RP_my")
        # observ_space = spaces.Box(low=np.array([300, 0, 300, 500]), high=np.array([550, 1, 350, 550]), dtype=np.int)

        if observation_space is None:
            observation_space = [[300, 0, 300, 500], [550, 210, 350, 550]]

        self.set_observation_space(observation_space=observation_space)

        # set the action space ....
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
            obs, [offer, time]
            Just return the my negotiator observation,
            while set the game as competition firstly, information less
        """
        obs: List = self.game.get_observation()

        return obs[0]

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
        for hybrid action space
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
import akro
from gym import spaces
from drl_negotiation.core.core import TrainWorld
from drl_negotiation.core.hyperparameters import *
from drl_negotiation.core._env import Environment, EnvSpec, EnvStep
from drl_negotiation.core._dtypes import StepType

class SCMLEnv(Environment):
    metadata = {
            'render.modes': ['human']
            }

    def __init__(
            self,
            world: Union[TrainWorld],
            goal=GOAL,
            reset_callback=None,
            reward_callback=None,
            observation_callback=None,
            info_callback=None,
            done_callback=None,
            shared_viewer=True,
            ):

        self._goal = goal
        self.world = world
        # trainable agents
        self.agents = self.world.policy_agents
        self.heuristic_agents = self.world.heuristic_agents

        # vectorized gym env property
        self.n = len(self.agents)
        self.extra_n = len(self.heuristic_agents)
        self._step_cnt = None
        self._visualize = False
        # callback
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # env parameters
        self.discrete_action_space = DISCRETE_ACTION_SPACE
        # action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = DISCRETE_ACTION_INPUT
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self._step_cnt = None
        self._max_episode_length = self.world.n_steps
        # spaces
        self._action_space = []
        self._observation_space = []
        for agent in self.agents:

            # for seller
            total_action_space = []
            # for buyer
            total_action_space_buyer = []

            # negotiation management action space
            m_action_space = []
            b_action_space = []
            consumers = len(agent.awi.my_consumers)
            suppliers = len(agent.awi.my_suppliers)

            if self.discrete_action_space:
                # seller controller
                for _ in range(consumers):
                    m_action_space.append(spaces.Discrete(world.Q))
                    m_action_space.append(spaces.Discrete(world.T))
                    m_action_space.append(spaces.Discrete(world.U))
                if not ONLY_SELLER:
                    # buyer controller
                    for _ in range(suppliers):
                        b_action_space.append(spaces.Discrete(world.Q))
                        b_action_space.append(spaces.Discrete(world.T))
                        b_action_space.append(spaces.Discrete(world.U))
            else:
                for _ in range(consumers):
                    m_action_space.append(spaces.Box(low=-agent.m_range, high=+agent.m_range, shape=(world.dim_m, ),
                                                     dtype=np.float32))
                if not ONLY_SELLER:
                    for _ in range(suppliers):
                        b_action_space.append(spaces.Box(low=-agent.b_range, high=+agent.b_range, shape=(world.dim_b,),
                                                         dtype=np.float32))

            if agent.manageable:
                total_action_space = m_action_space
                if not ONLY_SELLER:
                    total_action_space_buyer = b_action_space

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c, ), dtype=np.float32)

            if not agent.silent and c_action_space.n!=0:
                total_action_space.append(c_action_space)
                total_action_space_buyer.append(c_action_space)

            # for seller
            if len(total_action_space) >1:

                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    # act_space = spaces.MultiDiscrete([[0, act_space.n -1] for act_space in total_action_space])
                    act_space = spaces.MultiDiscrete([act_space.n for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self._action_space.append(act_space)
            else:
                self._action_space.append(total_action_space[0])

            # for buyer
            if len(total_action_space_buyer) > 1:
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space_buyer]):
                    # act_space = spaces.MultiDiscrete([[0, act_space.n -1] for act_space in total_action_space])
                    act_space = spaces.MultiDiscrete([act_space.n for act_space in total_action_space_buyer])
                else:
                    act_space = spaces.Tuple(total_action_space_buyer)
                self._action_space.append(act_space)
            else:
                self._action_space.append(total_action_space_buyer[0])

            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self._observation_space.append(akro.Box(low=-np.inf, high=+np.inf, shape=(obs_dim, ), dtype=np.float32))
            if not ONLY_SELLER:
                obs_dim = len(observation_callback(agent, self.world, seller=False))
                self._observation_space.append(akro.Box(low=-np.inf, high=+np.inf, shape=(obs_dim, ), dtype=np.float32))

            agent.action.c = np.zeros(self.world.dim_c)

        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=self._max_episode_length)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            # policy agents
            self.viewers = [None] * self.n
        self._reset_render()

    @property
    def possible_agents(self):
        if not ONLY_SELLER:
            return np.reshape(np.array([[f"{agent.agent_scope_name}_seller", f"{agent.agent_scope_name}_buyer"] for agent in self.agents]),
                              (1, 2*len(self.agents)))[0]
        else:
            return np.array([f"{agent.agent_scope_name}" for agent in self.agents])

    @property
    def action_space(self):
        """np.ndarray[akro.Space]: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """np.ndarray[akro.Space]: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return [
            'ascii',
        ]

    @property    
    def step_cnt(self):
        return self._step_cnt

    def reset(self):
        # reset world
        # self.reset_callback(self.world)
        self.world = self.reset_callback(self.world)

        first_obs = []
        self.agents = self.world.policy_agents
        # initial the state of agents

        for agent in self.agents:
            self.world.update_agent_state(agent)

        # and get the initial obs
        for agent in self.agents:
            first_obs.append(self._get_obs(agent, seller=True))
            if not ONLY_SELLER:
                first_obs.append(self._get_obs(agent, seller=False))
        self._step_cnt = 0
        return np.array(first_obs), dict(goal=self._goal)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        extra_rew = []
        self.agents = self.world.policy_agents
        
        # policy agents
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i*2], agent, self.action_space[i*2])
            if not ONLY_SELLER:
                # buyer action, the same action_space as the seller
                self._set_buyer_action(action_n[i*2+1], agent, self.action_space[i*2+1])

        self.world.step()
        for agent in self.agents:
            obs_n.append(self._get_obs(agent, seller=True))
            reward_n.append(self._get_reward(agent, seller=True))
            done_n.append(self._get_done(agent, seller=True))
            info_n['n'].append(self._get_info(agent, seller=True))

            if not ONLY_SELLER:
                obs_n.append(self._get_obs(agent, seller=False))
                reward_n.append(self._get_reward(agent, seller=False))
                done_n.append(self._get_done(agent, seller=False))
                info_n['n'].append(self._get_info(agent, seller=False))

            # update state after calculate reward
            agent.state.f[1] = agent.state.f[2]

        for agent in self.world.heuristic_agents:
            extra_rew.append(self._get_reward(agent, seller=True))
            if not ONLY_SELLER:
                extra_rew.append(self._get_reward(agent, seller=False))

        if RENDER_INFO:
            self.info_n = info_n

        # if not ONLY_SELLER:
        #     obs_n = obs_n * 2
        #     reward_n = reward_n * 2
        #     done_n = done_n * 2
        #     info_n['n'] = info_n['n'] * 2

        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
            if not ONLY_SELLER:
                reward_n *= 2
            # if not ONLY_SELLER:
            #     reward_n = reward_n * 2

        self._step_cnt +=1

        step_type = [StepType.get_step_type(
            step_cnt=self._step_cnt,
            max_episode_length=self._max_episode_length,
            done=_) for _ in done_n]

        if all([1 if s in (StepType.TERMINAL, StepType.TIMEOUT) else 0 for s in step_type]):
            self._step_cnt = None

        # print(f"reward_n are {reward_n}")
        return EnvStep(
            env_spec=self.spec,
            action=action_n,
            reward=np.array(reward_n),
            extra_rew=np.array(extra_rew),
            observation=np.array(obs_n),
            env_info=info_n,
            step_type=np.array(step_type)
        )

    def render(self, mode="ascii"):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.

        Returns:
            str: the point and goal of environment.

        """

        profit = [agent.state.f[2] - agent.state.f[0] for agent in self.agents]
        return f"profit: {sum(profit)}, Goal: {self._goal}"

    def visualize(self):
        """Creates a visualization of the environment."""

        self._visualize = True
        print(self.render('ascii'))

    def close(self):
        """Close the env."""

    def _reset_render(self):
        self.infos = None

    def _get_info(self, agent, seller=True):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world, seller=seller)

    def _get_obs(self, agent, seller=True):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world, seller=seller)

    def _get_done(self, agent, seller=True):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world, seller=seller)

    def _get_reward(self, agent, seller=True):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world, seller=seller)

    def _preprocess_action(self, action, action_space):
        if isinstance(action_space, spaces.MultiDiscrete):
            act = []
            index = 0
            if self.discrete_action_input:
                action = [action]
            else:
                for s in action_space.nvec:
                    act.append(action[index:(index+s)])
                    index +=s
                action = act
        else:
            action = [action]
        return action

    def _process_manager_action(self, agent, action, seller=True):
        if agent.manageable:
            #TODO: negotiation management action, both seller and buyer
            if self.discrete_action_input:
                if seller:
                    agent.action.m = np.zeros(self.world.dim_m * len(agent.awi.my_consumers))
                    for i in range(self.world.dim_m * len(agent.awi.my_consumers)):
                        agent.action.m[i] = action[0][i]
                else:
                    agent.action.b = np.zeros(self.world.dim_b * len(agent.awi.my_suppliers))
                    #process discrete action
                    for i in range(self.world.dim_b * len(agent.awi.my_suppliers)):
                        agent.action.b[i] = action[0][i]
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    if seller:
                        for i in range(self.world.dim_m):
                            agent.action.m[i] +=action[0][2*i+1] - action[0][2*(i+1)]
                    else:
                        for i in range(self.world.dim_b):
                            agent.action.b[i] +=action[0][2*i+1] - action[0][2*(i+1)]
                else:
                    if seller:
                        agent.action.m = action[0]
                    else:
                        agent.action.b = action[0]
            #ipdb.set_trace()
            action = action[1:]
        return action

    def _process_communication_action(self, agent, action):
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]

            action = action[1:]
        return action

    def _set_buyer_action(self, action, agent, action_space, time=None):
        # set the action of buyer
        agent.action.b = np.zeros(self.world.dim_b * len(agent.awi.my_suppliers))

        action = self._preprocess_action(action, action_space)
        self._process_manager_action(agent, action, seller=False)

    def _set_action(self, action, agent, action_space, time=None):
        # set the action of seller and communication
        agent.action.m = np.zeros(self.world.dim_m * len(agent.awi.my_consumers))
        agent.action.c = np.zeros(self.world.dim_c)
       
        # process action
        action = self._preprocess_action(action, action_space)
        action = self._process_manager_action(agent, action, seller=True)
        self._process_communication_action(agent, action)


