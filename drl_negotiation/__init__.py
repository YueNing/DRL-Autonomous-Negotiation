# Library for training negotiators in SAOMechanism and agents in SCML2020World

# Simulation Library
# negmas, http://yasserm.com/negmas/
# scml, http://www.yasserm.com/scml/scml2020docs/

# Two main modules
# Module 1: Train a Negotiator in SAOMechanism
# Idea comes from ANegma, https://arxiv.org/abs/2001.11785

# Module 2: Train an Agent in SCML2020World
# Idea comes from an algorithm called multi-agent DDPG proposed by OpenAI
# Multi-Agent Actor-Critic for MixedCooperative-Competitive Environments link: https://arxiv.org/pdf/1706.02275.pdf
# Extension of MADDPG, link: https://github.com/openai/maddpg/

name = "drl_negotiation"
author = "naodongbanana"
email = "n1085633848@outlook.com"
license = "MIT License"


import time
from drl_negotiation.utils import logging_setup

print("#################################### \n"
      "Welcome to drl negotiation, enjoy it!\n "
      "if you have any questions, redirect to\n"
      "uqveo@student.kit.edu\n"
      "####################################")

current_time = time.strftime('%a %d %b %Y %H:%M:%S +0000', time.localtime())
print(f"{current_time} Initial drl negotiation!")
logging_setup()
print("setup logging as ")