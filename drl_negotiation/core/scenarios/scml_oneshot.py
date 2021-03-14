import scml
from drl_negotiation.core.scenarios.scenario import BaseScenario
from drl_negotiation.core.utils.multi_agents_utils import generate_one_shot_world
from drl_negotiation.core.games.scml_oneshot import TrainWorld
from drl_negotiation.core.config.envs.scml_oneshot import *
from negmas.helpers import unique_name


class Scenario(BaseScenario):
    def info(self, agent: "Agent"):
        return agent.info()

    def done(self, agent: "Agent"):
        return agent.done()

    def reward(self, agent: "Agent"):
        if scml.__version__ == "0.3.1":
            agent = agent.controller
        return agent.reward()

    def observation(self, agent: "Agent"):
        # the real MyOneShotAgent is the controller of agent running in the SCMLOneShot
        # scml version 0.3.1, agent in RL is controller of  agent in environment
        if scml.__version__ == "0.3.1":
            agent = agent.controller
        return agent.observation()

    def reset_agent(self, agent: "Agent"):
        if agent.awi.current_step == 0:
            return
        return agent.reset()

    def make_world(self, config:dict = None) -> "World":
        world = generate_one_shot_world(
            ONESHOT_SCENARIO_02,
            n_processes=N_PROCESSES,
            name=unique_name(
                f"scml2020tests/single/{AGENT_TYPE[0].__name__}" f"Fine{N_PROCESSES}",
                add_time=True,
                rand_digits=4,
            ),
            compact=COMPACT,
            no_logs=NO_LOGS
        )
        return world

    def reset_world(self, world: "World") -> "ResetWorld":
        world.reset(world=self.make_world())
