from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from drl_negotiation.core._env import Environment


class RaySCMLEnv(MultiAgentEnv):
    """ An interface to the SCML MARL environment Library
    """

    def __init__(self, env: Environment):
        self.env = env
        # agent idx list
        self.agents = self.env.available_agents

        # get dictionaries of obs_spaces and act_spaces
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

        # Get first observation space, assuming all agents have equal space
        self.observation_space = self.env.observation_space

        # Get first action space, assuming all agents have equal space
        self.action_space = self.env.action_space

        self.reset()

    def reset(self):
        return self.env.reset()

    def step(self, action_dict):
        obs_d, rew_d, done_d, info_d = self.env.step(action_dict)

        return  obs_d, rew_d, done_d, info_d

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def render(self, mode="human"):
        return self.env.render(mode)