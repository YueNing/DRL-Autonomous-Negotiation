import numpy as np
import abc
from abc import ABC
from drl_negotiation.core._dtypes import Env
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelResult:
    """
    final_ep_rewards (numpy.ndarray):  rl agents mean reward of episode based on the save rate
    final_ep_extra_rewards (numpy.ndarray): others agents mean reward of episode in the SL2020World, competitive agents
                                        or cooperative agents.
    final_ep_ag_rewards (numpy.ndarray): rl agents mean reward of episode based on the save rate and agents
    final_ep_extra_ag_rewards (numpy.ndarray): other agents mean reward of episode in the SCML2020World
    episode_rewards (numpy.ndarray): every episode reward of rl agents
    episode_extra_rewards (numpy.ndarray): every episode reward of others agents running in the SCML2020World
    env: running environment
    """
    final_ep_rewards: np.ndarray
    final_ep_extra_rewards: np.ndarray
    final_ep_ag_rewards: np.ndarray
    final_ep_extra_ag_rewards: np.ndarray
    episode_rewards: np.ndarray
    episode_extra_rewards: np.ndarray
    env: Env

    @property
    def total_final_ep_rewards(self):
        return np.concatenate((self.final_ep_rewards, self.final_ep_extra_rewards))

    @property
    def total_final_ep_ag_rewards(self):
        return np.concatenate((self.final_ep_ag_rewards, self.final_ep_extra_ag_rewards))

    @property
    def total_episode_rewards(self):
        return np.concatenate((self.episode_rewards, self.episode_extra_rewards))


class Runner(ABC):

    def setup(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError