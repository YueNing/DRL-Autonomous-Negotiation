from drl_negotiation.agents.myagent import MyOneShotBasedAgent
import scml
if scml.__version__ == "0.3.1":
    from scml.oneshot.builtin import RandomOneShotAgent
else:
    from scml.oneshot.agents import RandomOneShotAgent

AGENT_TYPE = [MyOneShotBasedAgent]
ONESHOT_SCENARIO_01 = AGENT_TYPE
ONESHOT_SCENARIO_02 = AGENT_TYPE + [RandomOneShotAgent]

N_PROCESSES = 2
COMPACT = True
NO_LOGS = True

BATCH = None
