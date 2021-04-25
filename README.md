# Concurrent Negotiation Control with mulit-agent deep reinforcement learning
> ***Note:*** The final source code repository for thesis

**Question 1: Dynamical Range Of Negotiation Issues**
At the beginning of every negotiation in simulator, agent will determine the range which constraints value interval for
negotiation issues. In the experiment, the negotiation issues are **QUANTITY**, **PRICE** and
**TIME**. After creating the simulation world, simulator determines the minimum and maximum
values for each negotiation issue taken by the entire simulation episode, such as value of
**QUANTITY** between (1, 10), **PRICE** between (0, 100) and **TIME** between (0, 100). 
However, for every negotiation mechanism created inside the entire simulation episode, it has its dynamic range of negotiation issues which affect the negotiation process. This question was raised 
based on such a situation.

### Conditions

Condition 1: Set negotiated rate equal to the speed of the simulation world

Condition 2: Set disallow_concurrent_negs_with_same_partners as True

### MADDPG with Ray
Distributed training
