import abc
from abc import ABC


class TrainableAgent(ABC):

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def observation(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reward(self):
        raise NotImplementedError

    @abc.abstractmethod
    def info(self):
        raise NotImplementedError

    @abc.abstractmethod
    def done(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_action(self, action):
        raise NotImplementedError

    @abc.abstractmethod
    def policy(self, observation):
        raise NotImplementedError


class TrainingWorld(ABC):
    """Wrapper of World"""
    def __init__(self, world):
        self._world = world

    @property
    def world(self):
        return self._world

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError