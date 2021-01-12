##############################################################
# scml sccenario, for concurrent negotiation control
##############################################################
from drl_negotiation.scenario import BaseScenario
from drl_negotiation.hyperparameters import *
from drl_negotiation.utils import make_world
from scml import SCML2020World

class Scenario(BaseScenario):
    def make_world(self, config):
        # configuration, for Scenario scml_concurrent
        world = make_world(config=config)
        return world

    def reset_world(self, world):
        agent_types = world.configuration['agent_types']
        agent_params = world.configuration['agent_params'][:-2]
        n_steps = world.configuration['n_steps']

        reset_configuration = SCML2020World.generate(
            agent_types=agent_types,
            agent_params=agent_params,
            n_steps=n_steps
        )

        world.__init__(configuration=reset_configuration)

    def observation(self, agent, world, seller=True):
        _obs = []
        return _obs

    def reward(self):
        pass

    def done(self):
        pass

    def benchmark_data(self):
        pass