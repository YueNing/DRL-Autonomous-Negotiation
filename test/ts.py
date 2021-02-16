from drl_negotiation.third_party.scml.src.scml.scml2020 import SCML2020World, DecentralizingAgent, BuyCheapSellExpensiveAgent, IndDecentralizingAgent, MovingRangeAgent

def test_scml_world():
    agent_types = [
        DecentralizingAgent,
        BuyCheapSellExpensiveAgent,
        IndDecentralizingAgent,
        MovingRangeAgent
    ]

    world = SCML2020World(
        **SCML2020World.generate(
            agent_types=agent_types,
            n_steps=50
        )
    )

    world.draw()
    world.run_with_progress()
if __name__ == '__main__':
    test_scml_world()