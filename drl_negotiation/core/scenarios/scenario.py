import abc
from abc import ABC


class BaseScenario(ABC):
    """The main API for scml negotiation Scenario.

        The public API methods are:
        +-----------------------+
        | Functions             |
        +=======================+
        | make_world()          |
        +-----------------------+
        | reset_world()         |
        +-----------------------+
        | reset_agent()         |
        +-----------------------+

        Callback functions called by Environment
        +-----------------------+
        | observation()         |
        +-----------------------+
        | reward()              |
        +-----------------------+
        | done()                |
        +-----------------------+
        | info()                |
        +-----------------------+
    """

    @abc.abstractmethod
    def make_world(self, config: "WorldConfig") -> "World":
        """Creates instance of game or training world"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_world(self, world: "World") -> "ResetWorld":
        """Sets the world to the initial state"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset_agent(self, agent: "Agent"):
        """Resets agent, returns initial observation"""
        raise NotImplementedError

    @abc.abstractmethod
    def observation(self, agent: "Agent"):
        """callback observe by agent"""
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self, agent: "Agent"):
        """callback reward of agent"""
        raise NotImplementedError

    @abc.abstractmethod
    def done(self, agent: "Agent"):
        """callback done of agent"""
        raise NotImplementedError

    @abc.abstractmethod
    def info(self, agent: "Agent"):
        """callback info of agent"""
        raise NotImplementedError
