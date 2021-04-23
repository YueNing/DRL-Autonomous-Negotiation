# Design of RL-Agent in SCMLOneShot

# Builtin Agent Types
- scml.oneshot.agents.SyncRandomOneShotAgent
- scml.oneshot.agents.SingleAgreementRandomAgent
- scml.oneshot.agents.SingleAgreementAspirationAgent
- scml.oneshot.agents.GreedyOneShotAgent
- scml.oneshot.agents.GreedySyncAgent
- scml.oneshot.agents.GreedySingleAgreementAgent
- scml.oneshot.agents.OneshotDoNothingAgent

## Scenario-01

Issues: Quantity with range (1, 10), Unit_Price with range (1, 100)

Number of Agents are 2

MyOneShotBasedAgent cooperates with MyOneShotBasedAgent

Goal: Maximize the Score of agents whose type is MyOneShotBasedAgent

### Design of Observation (single agent)

- ~~my_last_offer: dimension is (10, 100), size of onehot encoding is 10*100~~
- ~~range of negotiation issue, dimension is~~
- current_offer: dimension is (10, 100), size of onehot encoding is 10*100 

### Design of State (single agent)

- ~~range of negotiation issue~~
- ~~negotiation step~~
- ~~simulation step~~
- observation instead state (current offer)

### Design of Action (single agent)

Action of every agent is: (quantity, unit_price), dimension is (10, 100), size of onehot action is 10*100

### Design of Reward (single agent)

Possible rewards

- ~~ufun in the negotiation process (return after every negotiation step)~~ 0
- utility function in the simulation process (return after every simulation step)

### Algorithm

- QMIX: Design of joint q-value

### Result
![my_co_my.png](https://i.loli.net/2021/03/09/alMn7rJ6sQfwcbP.png)

---------------------------------------------------------------

## Scenario-02

Issues: Quantity with range (1, 10), Unit_Price with range (1, 100)

Number of Agents are 2

MyOneShotBasedAgent vs RandomOneShotAgent

Goal: Maximize the Score of agents whose type is MyOneShotBasedAgent

### Design of Observation

- ~~my_last_offer: dimension is (10, 100), size of onehot encoding is 10*100~~
- current_offer: dimension is (10, 100), size of onehot encoding is 10*100

### Design of State

- observation instead state

### Design of Action 

Action of every agent is: (quantity, unit_price), dimension is (10, 100), size of onehot action is 10*100

### Design of Reward

Possible rewards

- ~~ufun in the negotiation process (return after every negotiation step)~~
- utility function in the simulation process (return after every simulation step)

### Design of joint q-value
- QMIX (in this scenario has just one Agent)

----------------------------------------------------------------------------------------------------

## Scenario-03

Issues: Quantity with range (1, 10), Unit_Price with range (1, 100)

Number of Agents are 4

Two MyOneShotBasedAgent vs Two RandomOneShotAgent

Goal: Maximize the Score of agents whose type is MyOneShotBasedAgent

### Design of Observation

- my_last_offer: dimension is (10, 100), size of onehot encoding is 10*100
- current_offer: dimension is (10, 100), size of onehot encoding is 10*100

### Design of State

- None

### Design of Action

Action of every agent is: (quantity, unit_price), dimension is (10, 100), size of onehot action is 10*100

### Design of Reward

Possible rewards

- ufun in the negotiation process (return after every negotiation step)
- utility function in the simulation process (return after every simulation step)

### Design of joint q-value
- QMIX

---------------------------------------------------------------------------------------------------------------------

## Others' Scenario

Many vs Many

Same as Scenario-03