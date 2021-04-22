### Autonomous Negotiation with Deep Reinforcement Learning(DRL)

> ***Note:*** The final source code repository for thesis

This project originated from @Yue's master's thesis @kit. The purpose of this research is to try to use innovative DRL methods to realize smart negotiators and smart factory managers.

It is mainly splited as two parts:

- Single Agent(Negotiator) autonomous negotiation under a specific negotiation mechanism
- Multi Agent(Factory Manager) autonomous concurrent negotiations under simulated supply chain management world

### Installation

Installation of Anaconda under ubuntu

Create new virtual python environment
```python==3.7```

Installation of requirments
```negmas, scml```

After installation, you can perform a test to check whether the installation is correct.

### Environments

**Single-Agent(Negotiator) Autonomous Negotiation**

In single negotiator scenario, two strategies called **acceptance strategy** and **offer strategy** are learned by DRL negotiator. For every strategy there are two specific settings are given: single negotiation issue and multi-issues negotiation.

Detailed source code can be found in [single agent(negotiator) autonomous negotiation](https://github.com/YueNing/summary_thesis/blob/master/summary-2020-10a11/2020-10a11-Ningyue-Negmas-negotiation.ipynb)

**Multi-Agent(Factory Manager) autonomous concurrent negotiations**

### Examples
See the _examples_ directory.

- run , result(mean reward of acceptacen strategy) of single agent(negotiator) autonomous neogtiation 
![single.PNG](https://i.loli.net/2021/04/23/ytPOCNMxLlSaTDh.png)

### Testing
It is using pytest for tests. You can run them via:
```pytest```

### Resources





