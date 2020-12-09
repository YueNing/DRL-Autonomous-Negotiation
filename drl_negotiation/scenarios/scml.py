from drl_negotiation.scenario import BaseScenario
from drl_negotiation.core import SCMLWorld

class Scenario(BaseScenario):

    def make_world(self):
        world = SCMLWorld()
        self.reset_world()
        return world

    def reset_world(self):
        pass

    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Difference from initial funds
        pass

    def adversary_reward(self, agent, world):
        # keep the good agents near the intial funds
        # neg reward
        # pos reward
        rew = 0
        pass

    def observation(self, agent, world):
        # get all 
        pass
