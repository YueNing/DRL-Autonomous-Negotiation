import numpy as np
from typing import Optional, Union, List, Tuple
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
    def action_space(self, type=list):
        """np.ndarray[akro.Space]: The action space specification."""
        return self._action_space

    @property
    def observation_space(self, type=list):
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

    def get_action_space(self, type=list):
        """np.ndarray[akro.Space]: The action space specification."""
        if type == dict:
            return {f"{agent}": self._action_space[index] for index, agent in enumerate(self.possible_agents)}
        return self.action_space

    def get_observation_space(self, type=list):
        """np.ndarray[akro.Space]: The observation space specification."""
        if type == dict:
            return {f"{agent}": self._observation_space[index] for index, agent in enumerate(self.possible_agents)}
        return self.observation_space

    def reset(self):
        # reset world
        self.reset_callback(self.world)

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

import supersuit
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from drl_negotiation.core.envs.normalized_env import NormalizedEnv

class RaySCMLEnv(MultiAgentEnv):
    """An interface to the SCML MARL environment library.
    """

    def __init__(self, env: NormalizedEnv):
        self.env = env

        # agent idx list
        self.agents = self.env.possible_agents

        # get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.env.get_observation_space(type=dict)
        self.action_spaces = self.env.get_action_space(type=dict)

        self.reset()

    def step(self, action_dict):
        action_n = list(action_dict.values())
        print(f"action_n are {action_n}")
        es = self.env.step(action_n)

        # make all list into dict
        dones = self._make_dict(es.last.tolist())
        if all(dones):
            dones['__all__'] = True
        else:
            dones['__all__'] = False

        result = (self._make_dict(es.observation.tolist()), \
               self._make_dict(es.reward.tolist()), dones, \
               self._make_dict(es.env_info))

        return result

    def _make_dict(self, data):
        """
        convert list data to dict based on agents
        Args:
            data:

        Returns:

        """
        if type(data) == list:
            return {key: value for key, value in zip(self.agents, data)}
        elif type(data) == dict:
            return {key: value for key, value in zip(self.agents, data['n'])}
        else:
            raise NotImplementedError


    def reset(self):
        obs, _ = self.env.reset()

        _dict_obs = {}
        for i in range(len(self.agents)):
            _dict_obs[self.agents[i]] = obs[i]

        return _dict_obs
