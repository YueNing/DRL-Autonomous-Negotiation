from abc import ABC
from drl_negotiation.core.games._game import TrainableAgent
from drl_negotiation.third_party.scml.src.scml.oneshot.agent import OneShotAgent


class MyOneShotAgent(OneShotAgent, TrainableAgent, ABC):
    """Running in the SCML OneShot"""
    def __init__(self, policy_callback=None):
        super(MyOneShotAgent, self).__init__()
        self.policy_callback = policy_callback

    def done(self):
        pass

    def info(self):
        pass

    def reward(self):
        pass

    def observation(self):
        pass

    def reset(self):
        pass

    def set_action(self, action):
        pass

    def policy(self, observation):
        """return the action, get policy_callback from the trainer"""
        self.policy_callback(observation)


class Agents:
    """Trained Agents, relates to Agents running in the SCML OneShot"""
    def __init__(self, args):
        self.n_actions = args.n_actions