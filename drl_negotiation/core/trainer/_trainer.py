import abc
from abc import ABC


class AgentTrainer(ABC):
    def __init__(self, name, model, obs_shape, act_space, args):
        raise NotImplemented()

    @abc.abstractmethod
    def action(self, obs):
        raise NotImplemented()

    @abc.abstractmethod
    def experience(self, obs, act, rew, new_obs, done, terminal):
        raise NotImplemented()

    @abc.abstractmethod
    def preupdate(self):
        raise NotImplemented()

    @abc.abstractmethod
    def update(self, agent, t):
        raise NotImplemented()