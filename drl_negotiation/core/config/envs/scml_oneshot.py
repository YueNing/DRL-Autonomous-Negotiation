from drl_negotiation.agents.myagent import MyOneShotBasedAgent
from scml.oneshot.agents import (
RandomOneShotAgent,
    SyncRandomOneShotAgent,
    SingleAgreementRandomAgent,
    SingleAgreementAspirationAgent,
    GreedyOneShotAgent,
    GreedySyncAgent,
    GreedySingleAgreementAgent,
    OneshotDoNothingAgent
)

AGENT_TYPE = [MyOneShotBasedAgent, MyOneShotBasedAgent]
ONESHOT_SCENARIO_01 = AGENT_TYPE
ONESHOT_SCENARIO_02 = AGENT_TYPE + [GreedyOneShotAgent, GreedySyncAgent]

N_PROCESSES = 2
N_AGENTS_PER_PROCESS = 2
COMPACT = True
NO_LOGS = True

BATCH = None
