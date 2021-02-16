"""Data types for agent-based learning."""
import enum
import typing
from drl_negotiation.core.games._scml import MySCML2020Agent
from drl_negotiation.core.envs.multi_agents_scml import SCMLEnv

class StepType(enum.IntEnum):
    """Defines the status of a :class:`~TimeStep` within a sequence.
    Note that the last :class:`~TimeStep` in a sequence can either be
    :attribute:`StepType.TERMINAL` or :attribute:`StepType.TIMEOUT`.
    Suppose max_episode_length = 5:
    * A success sequence terminated at step 4 will look like:
        FIRST, MID, MID, TERMINAL
    * A success sequence terminated at step 5 will look like:
        FIRST, MID, MID, MID, TERMINAL
    * An unsuccessful sequence truncated by time limit will look like:
        FIRST, MID, MID, MID, TIMEOUT
    """
    # Denotes the first :class:`~TimeStep` in a sequence.
    FIRST = 0
    # Denotes any :class:`~TimeStep` in the middle of a sequence (i.e. not the
    # first or last one).
    MID = 1
    # Denotes the last :class:`~TimeStep` in a sequence that terminates
    # successfully.
    TERMINAL = 2
    # Denotes the last :class:`~TimeStep` in a sequence truncated by time
    # limit.
    TIMEOUT = 3

    @classmethod
    def get_step_type(cls, step_cnt, max_episode_length, done):
        """Determines the step type based on step cnt and done signal.
        Args:
            step_cnt (int): current step cnt of the environment.
            max_episode_length (int): maximum episode length.
            done (bool): the done signal returned by Environment.
        Returns:
            StepType: the step type.
        Raises:
            ValueError: if step_cnt is < 1. In this case a environment's
            `reset()` is likely not called yet and the step_cnt is None.
        """
        if max_episode_length is not None and step_cnt >= max_episode_length:
            return StepType.TIMEOUT
        elif done:
            return StepType.TERMINAL
        elif step_cnt == 1:
            return StepType.FIRST
        elif step_cnt < 1:
            raise ValueError('Expect step_cnt to be >= 1, but got {} '
                             'instead. Did you forget to call `reset('
                             ')`?'.format(step_cnt))
        else:
            return StepType.MID


# Type of Agent
Agent: typing.Union[MySCML2020Agent]

# Represents a generic identifier for an agent (e.g., "agent1").
AgentID: typing.Any

# A dict keyed by agent ids, e.g. {"agent-1": value}.
MultiAgentDict = typing.Dict[AgentID, typing.Any]

# All Environments
Env = typing.Union[SCMLEnv]
