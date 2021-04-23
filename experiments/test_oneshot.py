import random

from scml.oneshot import SCML2020OneShotWorld
from scml.oneshot import builtin_agent_types
from scml.oneshot.agents import RandomOneShotAgent

random.seed(0)

def test_generate():
    world = SCML2020OneShotWorld(
        **SCML2020OneShotWorld.generate(
            agent_types=RandomOneShotAgent,
            n_agents_per_process=1,
            n_steps=10,
            n_processes=2,
            n_lines=10,
        )
    )
    world.run()
    assert True

def test_agent():
    for agent_type in builtin_agent_types(as_str=True):
        print(agent_type)

if __name__ == '__main__':
    test_agent()

