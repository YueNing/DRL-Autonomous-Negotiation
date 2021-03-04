from drl_negotiation.agents.myagent import MyOneShotBasedAgent
from scml.oneshot.builtin import RandomOneShotAgent

AGENT_TYPE = [MyOneShotBasedAgent]
ONESHOT_SCENARIO_01 = AGENT_TYPE
ONESHOT_SCENARIO_02 = AGENT_TYPE + [RandomOneShotAgent]

N_PROCESSES = 2
COMPACT = True
NO_LOGS = True

BATCH = None
