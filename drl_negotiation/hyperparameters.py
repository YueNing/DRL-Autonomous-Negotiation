################################################
# Global
###############################################
import logging

# Training from scratch: set TRAIN, SAVE_WORLD_CONFIG as True
# Training from checkpoints: set TRAIN as True, SAVE_WORLD_CONFIG as False
# Evaluation from trained model: set TRAIN, SAVE_WORLD_CONFIG as False
TRAIN = True
# train from checkpoints
RESTORE = True

LOAD_MODEL = False
LOGGING_LEVEL = logging.INFO
FILENAME = "my.log"
EVALUATION = not TRAIN
# if train, not restore, train from scratch
SAVE_WORLD_CONFIG = True

if RESTORE and TRAIN:
    SAVE_WORLD_CONFIG = False
if EVALUATION:
    SAVE_WORLD_CONFIG = False

LOAD_WORLD_CONFIG = not SAVE_WORLD_CONFIG

SAVE_WORLD_CONFIG_DIR = "./world.config"
LOAD_WORLD_CONFIG_DIR = " "
TRAIN_EPISODES = 10

if LOAD_WORLD_CONFIG_DIR == " ":
    LOAD_WORLD_CONFIG_DIR = SAVE_WORLD_CONFIG_DIR

# Train only the seller component of agent
ONLY_SELLER = False # train seller and buyer together
SAVE_DIR = "/tmp/policy/"
MODEL_NAME = "model"
SAVE_RATE = 1

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
RENDER_INFO = True

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
