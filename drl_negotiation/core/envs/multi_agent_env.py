import abc
from abc import ABC
from drl_negotiation.core._dtypes import MultiAgentDict, AgentID
from drl_negotiation.core._env import Environment


class MultiAgentEnv(Environment, ABC):
    """The main API for multi agents negotiation environments.

        The public API methods are:
        +-----------------------+
        | Functions             |
        +=======================+
        | reset()               |
        +-----------------------+
        | step()                |
        +-----------------------+
        | render()              |
        +-----------------------+
        | close()               |
        +-----------------------+
        | get_obs()             |
        +-----------------------+
        | get_obs_agent()       |
        +-----------------------+
        | get_obs_size()        |
        +-----------------------+
        | get_state()           |
        +-----------------------+
        | get_state_size()      |
        +-----------------------+
        | get_avail_actions()   |
        +-----------------------+
        | get_avail_agent_      |
        | actions()             |
        +-----------------------+

        Set the following properties:

        +-----------------------+-------------------------------------------------+
        | Properties            | Description                                     |
        +=======================+=================================================+
        | action_space          | The action space specification                  |
        +-----------------------+-------------------------------------------------+
        | observation_space     | The observation space specification             |
        +-----------------------+-------------------------------------------------+
        | spec                  | The environment specifications                  |
        +-----------------------+-------------------------------------------------+
        | render_modes          | The list of supported render modes              |
        +-----------------------+-------------------------------------------------+
        | seed                  | The seed of environment                         |
        +-----------------------+-------------------------------------------------+

        Example of a simple rollout loop:
    """

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
        """list: A list of string representing the supported render modes.
        See render() for a list of modes.
        """
        return self._render_modes

    @property
    def seed(self):
        return self._seed

    @abc.abstractmethod
    def step(self, action_dict: MultiAgentDict):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    @abc.abstractmethod
    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    @abc.abstractmethod
    def get_obs_agent(self, agent_id: AgentID):
        """ Returns observation for agent_id """
        raise NotImplementedError

    @abc.abstractmethod
    def get_rew_agent(self, agent_id: AgentID):
        raise NotImplementedError

    @abc.abstractmethod
    def get_done_agent(self, agent_id: AgentID):
        raise NotImplementedError

    @abc.abstractmethod
    def get_info_agent(self, agent_id: AgentID):
        raise NotImplementedError

    @abc.abstractmethod
    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_avail_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_avail_agent_actions(self, agent_id: AgentID):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

    @abc.abstractmethod
    def save_replay(self):
        raise NotImplementedError

    def visualize(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
