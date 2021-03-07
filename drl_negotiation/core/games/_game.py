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
        self._rl_runner = None

    @property
    def world(self):
        return self._world

    @property
    def rl_runner(self):
        return self._rl_runner

    @rl_runner.setter
    def rl_runner(self, runner):
        self._rl_runner = runner

    @property
    def rollout_worker(self) -> "RolloutWorker":
        return self.rl_runner.rollout_worker

    @rollout_worker.setter
    def rollout_worker(self, worker):
        self.rollout_worker = worker

    @property
    def env(self) -> "MultiNegotiationSCM":
        return self.rl_runner.env

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
