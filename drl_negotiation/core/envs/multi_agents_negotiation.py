import random
import numpy as np
from drl_negotiation.core.envs.multi_agent_env import MultiAgentEnv
from drl_negotiation.core._dtypes import AgentID


class MutliNegotiationSCM(MultiAgentEnv):
    def __init__(self,
                 world,
                 reset_world_callback=None,
                 reset_agent_callback=None,
                 reward_callback=None,
                 observation_callback=None,
                 info_callback=None,
                 done_callback=None,
                 seed=None, ):
        # arguments
        self.world = world
        self.agents = self.world.policy_agents

        # callback
        self.reset_world_callback = reset_world_callback
        self.reset_agent_callback = reset_agent_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        # env arguments
        self.dones = set()
        self.resetted = False

        # spaces
        self._observation_space = None
        self._action_space = None

        # other
        self._max_episode_length = None
        self._seed = seed
        self._spec = None
        self._render_modes = ["ascii"]

        # setting
        random.seed(self.seed)

    def step(self, action_dict):
        assert len(self.dones) != len(self.agents)
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            self.agents[i].step(action)

        # world go one step, one step negotiation
        self.world.step()

        for i, action in action_dict.items():
            obs[i] = self.get_obs_agent(i)
            rew[i] = self.get_rew_agent(i)
            done[i] = self.get_done_agent(i)
            info[i] = self.get_info_agent(i)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

    def reset(self):
        self.resetted = True
        self.dones = set()
        self.world = self.reset_world_callback(self.world)
        self.agents = self.world.policy_agents
        obs_dict = {i: self.reset_agent_callback(a) for i, a in enumerate(self.agents)}
        return obs_dict

    def get_obs(self):
        """ Returns all agent observations in a list.
        Note: Agents should have access only to their local
        observations during decentralised execution"""
        agents_obs = [self.get_obs_agent(_) for _ in range(self.agents)]
        return agents_obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(self.agents[agent_id])

    def get_rew_agent(self, agent_id: AgentID):
        if self.reward_callback is None:
            return 0
        return self.reward_callback(self.agents[agent_id])

    def get_done_agent(self, agent_id: AgentID):
        if self.done_callback is None:
            return random.choice([True, False])
        return self.done_callback(self.agents[agent_id])

    def get_info_agent(self, agent_id: AgentID):
        if self.info_callback is None:
            return {}
        return self.info_callback(self.agents[agent_id])
