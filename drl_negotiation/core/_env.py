import abc
import akro
import numpy as np
from dataclasses import dataclass
from typing import Dict
from drl_negotiation.core._dtypes import StepType


class Environment(abc.ABC):
    """The main API for negotiation environments.

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
    | visualize()           |
    +-----------------------+
    | close()               |
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

    Example of a simple rollout loop:
    """

    @property
    @abc.abstractmethod
    def action_space(self):
        """np.ndarray[akro.Space]: The action space specification."""

    @property
    @abc.abstractmethod
    def observation_space(self):
        """np.ndarray[akro.Space]: The observation space specification."""

    @property
    @abc.abstractmethod
    def observation_spaces(self):
        """np.ndarray[gym.Space]: Multi agents, the observation spaces specification
            just set up in multi agents environment
        """

    @property
    @abc.abstractmethod
    def action_spaces(self):
        """np.ndarray[gym.Space]: Multi agents, the action spaces specification
            just set up in multi agents environment
        """

    @property
    @abc.abstractmethod
    def spec(self):
        """EnvSpec: The environment specification."""

    @property
    @abc.abstractmethod
    def render_modes(self):
        """list: A list of string representing the supported render modes.
        See render() for a list of modes.
        """

    @property
    @abc.abstractmethod
    def available_agents(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def agent_selection(self):
        raise NotImplementedError

    @property
    def get_obs(self, agent):
        raise NotImplementedError

    @property
    def agents(self):
        raise NotImplementedError

    @abc.abstractmethod
    def seed(self, seed):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Resets the environment.
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """

    @abc.abstractmethod
    def step(self, action):
        """Steps the environment with the action and returns a `EnvStep`.
        If the environment returned the last `EnvStep` of a sequence (either
        of type TERMINAL or TIMEOUT) at the previous step, this call to
        `step()` will start a new sequence and `action` will be ignored.
        If `spec.max_episode_length` is reached after applying the action
        and the environment has not terminated the episode, `step()` should
        return a `EnvStep` with `step_type==StepType.TIMEOUT`.
        If possible, update the visualization display as well.
        Args:
            action (object): A NumPy array, or a nested dict, list or tuple
                of arrays conforming to `action_space`.
        Returns:
            EnvStep: The environment step resulting from the action.
        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.
        """

    @abc.abstractmethod
    def render(self, mode):
        """Renders the environment.
        The set of supported modes varies per environment. By convention,
        if mode is:
        * rgb_array: Return an `numpy.ndarray` with shape (x, y, 3) and type
            uint8, representing RGB values for an x-by-y pixel image, suitable
            for turning into a video.
        * ansi: Return a string (str) or `StringIO.StringIO` containing a
            terminal-style text representation. The text can include newlines
            and ANSI escape sequences (e.g. for colors).
        Make sure that your class's `render_modes` includes the list of
        supported modes.
        For example:
        .. code-block:: python
            class MyEnv(Environment):
                def render_modes(self):
                    return ['rgb_array', 'ansi']
                def render(self, mode):
                    if mode == 'rgb_array':
                        return np.array(...)  # return RGB frame for video
                    elif mode == 'ansi':
                        ...  # return text output
                    else:
                        raise ValueError('Supported render modes are {}, but '
                                         'got render mode {} instead.'.format(
                                             self.render_modes, mode))
        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.
        """

    @abc.abstractmethod
    def visualize(self):
        """Creates a visualization of the environment.
        This function should be called **only once** after `reset()` to set up
        the visualization display. The visualization should be updated
        when the environment is changed (i.e. when `step()` is called.)
        Calling `close()` will deallocate any resources and close any
        windows created by `visualize()`. If `close()` is not explicitly
        called, the visualization will be closed when the environment is
        destructed (i.e. garbage collected).
        """

    @abc.abstractmethod
    def close(self):
        """Closes the environment.
        This method should close all windows invoked by `visualize()`.
        Override this function in your subclass to perform any necessary
        cleanup.
        Environments will automatically `close()` themselves when they are
        garbage collected or when the program exits.
        """

    def _validate_render_mode(self, mode):
        if mode not in self.render_modes:
            raise ValueError('Supported render modes are {}, but '
                             'got render mode {} instead.'.format(
                self.render_modes, mode))

    def __del__(self):
        """Environment destructor."""
        self.close()


class Wrapper(Environment):
    """A wrapper for an environment that implements the `Environment` API."""

    def __init__(self, env):
        """Initializes the wrapper instance.
        Args:
            env (Environment): The environment to wrap
        """
        self._env = env

    def __getattr__(self, name):
        """Forward getattr request to wrapped environment.
        Args:
            name (str): attr (str): attribute name
        Returns:
             object: the wrapped attribute.
        Raises:
            AttributeError: if the requested attribute is a private attribute,
            or if the requested attribute is not found in the
            wrapped environment.
        """
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        if not hasattr(self._env, name):
            raise AttributeError('Attribute {} is not found'.format(name))
        return getattr(self._env, name)

    @property
    def action_space(self):
        """gym.Space: The action space specification."""
        return self._env.action_space

    @property
    def observation_space(self):
        """gym.Space: The observation space specification."""
        return self._env.observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._env.spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._env.render_modes

    def step(self, action):
        """Step the wrapped env.
        Args:
            action (np.ndarray): actions provided by agents.
        Returns:
            EnvStep: The environment step resulting from actions.
        """
        return self._env.step(action)

    def reset(self):
        """Reset the wrapped env.
        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)
        """
        return self._env.reset()

    def render(self, mode):
        """Render the wrapped environment.
        Args:
            mode (str): the mode to render with. The string must be
                present in `self.render_modes`.
        Returns:
            object: the return value for render, depending on each env.
        """
        return self._env.render(mode)

    def visualize(self):
        """Creates a visualization of the wrapped environment."""
        self._env.visualize()

    def close(self):
        """Close the wrapped env."""
        self._env.close()

    @property
    def unwrapped(self):
        """garage.Environment: The inner environment."""
        return getattr(self._env, 'unwrapped', self._env)


@dataclass(frozen=True)
class InOutSpec:
    """Describes the input and output spaces of a primitive or module."""
    input_space: akro.Space
    output_space: akro.Space


@dataclass(frozen=True, init=False)
class EnvSpec(InOutSpec):
    """Describes the observations, actions, and time horizon of an MDP.

    Args:
    observation_space (akro.Space): The observation space of the env.
    action_space (akro.Space): The action space of the env.
    max_episode_length (int): The maximum number of steps allowed in an
        episode.
    """
    def __init__(self,
                 observation_space,
                 action_space,
                 max_episode_length=None):
        object.__setattr__(self, 'max_episode_length', max_episode_length)
        super().__init__(input_space=action_space, output_space=observation_space)

    max_episode_length: int or None = None

    @property
    def action_space(self):
        """Get action space.

        Returns:
            akro.Space: Action space of the env.
        """
        return self.input_space

    @property
    def observation_space(self):
        """Get observation space of the env

        Returns:
            akro.Space: Observation space.
        """
        return self.output_space


@dataclass
class EnvStep:
    r"""A tuple representing a single step returned by the environment.
    Attributes:
        env_spec (EnvSpec): Specification for the environment from which this data was sampled
        action: (numpy.ndarray): A numpy array of shape :math:`(A^*)` containing the action for the this
            time step. These must conform to :obj: `EnvStep.action_space`.
        reward (numpy.ndarray): A float representing the reward for taking the action given the observation, at this
        time step.
        extra_rew (numpy.ndarray): reward of not trainable agents in the system
        observation (numpy.ndarray]): A numpy array of shape :math: `(O^*)`
            containing the observation for the this time step in the environment. These must conform to
            :obj: `EnvStep.observation_space`. The observation after applying the action.
        env_info: A dict containing environment state information
        done (numpy.ndarray(StepType)): A numpy array of shape :math: `(number of Agents)`, dtype is `StepType`
            `StepType` enum value, Can  either be StepType.FIRST, StepType.MID, StepType.TERMINAL, StepType.TIMEOUT.
    """
    env_spec: EnvSpec
    action: np.ndarray
    reward: np.ndarray
    extra_rew: np.ndarray
    observation: np.ndarray
    env_info: Dict[str, np.ndarray or dict]
    step_type: np.ndarray

    @property
    def first(self):
        """np.ndarray[bool]: Whether this `TimeStep` is the first of a sequence."""
        return np.array([StepType(s) is StepType.FIRST for s in self.step_type])

    @property
    def mid(self):
        """np.ndarray[bool]: Whether this `TimeStep` is the mid of a sequence."""
        return np.array([StepType(s) is StepType.MID for s in self.step_type])

    @property
    def terminal(self):
        """np.ndarray[bool]: Whether this `TimeStep` is the terminal of a sequence."""
        return np.array([StepType(s) is StepType.TERMINAL for s in self.step_type])

    @property
    def timeout(self):
        """np.ndarray[bool]: Whether this `TimeStep` is the timeout of a sequence."""
        return np.array([StepType(s) is StepType.TIMEOUT for s in self.step_type])

    @property
    def last(self):
        """np.ndarray[bool]: Whether this `TimeStep` is the timeout of a sequence."""
        return np.array([StepType(s) is StepType.TIMEOUT or s is StepType.TERMINAL for s in self.step_type])