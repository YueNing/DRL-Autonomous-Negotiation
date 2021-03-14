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

AGENT_TYPE = [MyOneShotBasedAgent]
ONESHOT_SCENARIO_01 = AGENT_TYPE
ONESHOT_SCENARIO_02 = AGENT_TYPE + [GreedyOneShotAgent]

N_PROCESSES = 2
COMPACT = True
NO_LOGS = True

BATCH = None
