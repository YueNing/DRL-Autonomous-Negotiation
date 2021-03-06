from drl_negotiation.core.games._game import TrainingWorld
from scml.oneshot import SCML2020OneShotWorld
from scml.scml2020 import is_system_agent
from drl_negotiation.core.utils.multi_agents_utils import generate_one_shot_world
from drl_negotiation.core.config.envs.scml_oneshot import *

__all__ = [
    "TrainWorld"
]


class TrainWorld(TrainingWorld):
    def __init__(self, world: SCML2020OneShotWorld):
        super(TrainWorld, self).__init__(world)

    @property
    def policy_agents(self):
        agents = {}
        for id, agent in self.world.agents.items():
            if is_system_agent(id):
                pass
            else:
                agents[id] = agent
        return agents

    def reset(self):
        """TODO: reset the world"""
        pass

    def step(self):
        self.world.step()

    def run(self):
        # result = self.world.run(self._rl_runner)
        result = self.world.run()
        return result



