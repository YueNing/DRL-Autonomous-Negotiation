import random
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from drl_negotiation.core.envs.multi_agent_env import MultiAgentEnv
# from drl_negotiation.core.config.envs.scml_oneshot import BATCH
from drl_negotiation.core.games._scml_oneshot import Agents


class MultiNegotiationSCM(MultiAgentEnv):
    """Focus on Multi Negotiation
    """
    @property
    def agents(self):
        return self._agents

    @agents.setter
    def agents(self, agents):
        self._agents = agents

    @property
    def agent_selection(self):
        pass

    @property
    def action_spaces(self):
        spaces = {}
        for id, agent in self.agents.items():
            spaces[id] = self.action_space
        return spaces

    @property
    def observation_spaces(self):
        spaces = {}
        for id, agent in self.agents.items():
            spaces[id] = self.observation_space
        return spaces

    def get_obs_size(self):
        return 10 * 100

    def get_state(self):
        if self.obs_instead_of_state:
            obs = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs

    def get_state_size(self):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        pass

    def get_avail_agent_actions(self, agent_id: "AgentID", issues):
        # Decide the available agent actions based on the
        # range of negotiation issues set by system
        from scml.oneshot import QUANTITY, UNIT_PRICE

        avail_actions = [0] * self.n_actions
        for i in range(issues[QUANTITY].values[0], issues[QUANTITY].values[1] + 1):
            for j in range(issues[UNIT_PRICE].values[0], issues[UNIT_PRICE].values[1] + 1):
                avail_actions[(i-1)*100 + (j-1)] = 1

        return avail_actions

    def get_total_actions(self):
        return self.n_actions

    def render(self):
        pass

    def close(self):
        pass

    def save_replay(self):
        pass

    def available_agents(self):
        pass

    action_space = Discrete(10 * 100)

    def __init__(self,
                 world=None,
                 scenario=None,
                 seed=None, ):
        # arguments
        self.world = world
        self.agents = self.world.policy_agents
        self.n_agents = len(self.agents)
        
        # callback
        self.reset_world_callback = scenario.reset_world
        self.reset_agent_callback = scenario.reset_agent
        self.reward_callback = scenario.reward
        self.observation_callback = scenario.observation
        self.info_callback = scenario.info
        self.done_callback = scenario.done

        # env arguments
        self.dones = set()
        self.resetted = False
        self.obs_instead_of_state = True
        self.episode_limit = self.world.world.n_steps * self.world.world.neg_n_steps

        # spaces
        self._observation_space = Discrete(10 * 100)
        self._action_space = Discrete(10 * 100)

        # other
        self._max_episode_length = None
        self._seed = seed
        self._spec = None
        self._render_modes = ["ascii"]

        # setting
        random.seed(self.seed)

        self._action_space = Discrete(10 * 100)
        self._observation_space = Discrete(10 * 100)

        self.n_actions = 10 * 100

    def step(self):
        self.world.step()
        if self.world.world.time > self.world.world.time_limit:
            return False
        if self.world.world.current_step >= self.world.world.n_steps:
            return False
        else:
            return True

    def run(self):
        self.world._rl_runner = self._rl_runner
        result = self.world.run()

        self.batch = None
        episode_result = self.batch
        return episode_result

    def reset(self):
        self.resetted = True
        self.dones = set()
        self.reset_world_callback(self.world)
        self.agents = self.world.policy_agents
        # self.world.step()
        obs_dict = {i: self.reset_agent_callback(a) for i, a in self.agents.items()}
        return obs_dict

    def get_obs(self):
        """ Returns all agent observations in a list.
        Note: Agents should have access only to their local
        observations during decentralised execution"""
        agents_obs = [self.get_obs_agent(_) for _ in self.agents]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(self.agents[agent_id])

    def get_reward(self):
        reward_n = [self.get_rew_agent(agent) for agent in self.agents]
        return np.sum(reward_n)

    def get_rew_agent(self, agent_id: "AgentID"):
        if self.reward_callback is None:
            return 0
        return self.reward_callback(self.agents[agent_id])

    def get_done_agent(self, agent_id: "AgentID"):
        if self.done_callback is None:
            return random.choice([True, False])
        return self.done_callback(self.agents[agent_id])

    def get_info_agent(self, agent_id: "AgentID"):
        if self.info_callback is None:
            return {}
        return self.info_callback(self.agents[agent_id])
