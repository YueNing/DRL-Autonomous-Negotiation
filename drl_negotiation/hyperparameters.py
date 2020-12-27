################################################
# Global
###############################################
import logging

TRAIN = True
LOAD_MODEL = False
LOGGING_LEVEL = logging.INFO
FILENAME = "my.log"

# Train only the seller component of agent
ONLY_SELLER = False # train seller and buyer together

################################################
# agent
################################################
MANAGEABLE = True
SLIENT = True
BLIND = False

###############################################
# SCML environment
###############################################
# action is a number 0...N, otherwise action is a one-hot N-dimensional vector
DISCRETE_ACTION_INPUT = False
DISCRETE_ACTION_SPACE = True

################################################
# negotiation mdoel
################################################
NEG_SELL_PATH = " "
NEG_BUY_PATH = None

################################################
# training world
################################################
DIM_M = 2
DIM_B = 2
TRAINING_AGENT_TYPES = ["drl_negotiation.myagent.MyComponentsBasedAgent", "scml.scml2020.DecentralizingAgent"]
REW_FACTOR = 0.2

################################################
# scml scenario
################################################

# used in SCML2020World, considered as days
N_STEPS = 10

################################################
# running world
################################################
#RUNNING_AGENT_TYPES = TRAINING_AGENT_TYPES
DIM_S = 2
RUNNING_AGENT_TYPES = TRAINING_AGENT_TYPES