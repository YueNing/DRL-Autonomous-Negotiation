################################################
# Global
###############################################
import logging
GOAL = 1000
# control the negotiation manager
# now running in scml2020world could just under condition, not train
RUNNING_IN_SCML2020World = False
POLICIES = ['/tmp/policy2/', '/tmp/policy3/']
# action of count all in controller
RANDOM = False
RANDOM_REWARD = False

# Training from scratch: set TRAIN, SAVE_WORLD_CONFIG as True
# Training from checkpoints: set TRAIN as True, SAVE_WORLD_CONFIG as False
# Evaluation from trained model: set TRAIN, SAVE_WORLD_CONFIG as False
TRAIN = True
# train from checkpoints
RESTORE = False
# Train only the seller component of agent
ONLY_SELLER = False # train seller and buyer together
# root dir, save all policies here
ROOT_DIR = "/" + "tmp" + "/"
# save dir, single policy
SAVE_DIR = ROOT_DIR+"policy4" + "/"
PLOTS_DIR = SAVE_DIR + "learning_curves" + "/"
# sns or plotly
PLOT_BACKEND = "plotly"
# running based on ray, create multi layer charts
MULTI_LAYER = True

# settings of plotly, online or local
ONLINE = False
USERNAME = "naodongbanana"
API_KEY = "kF579m9M6nXOrjw4GmKU"

# model name
MODEL_NAME = "model"
# train episode
TRAIN_EPISODES = 100
# train episode save rate
SAVE_RATE = 10
# max length of single episode
MAX_EPISODE_LEN = 25
# batch size
BATCH_SIZE = 16
# save the model of train separately
SAVE_TRAINERS = True

LOAD_MODEL = False
LOGGING_LEVEL = logging.INFO
LOGGING_FILE_LEVEL = logging.DEBUG
DISABLE_LOGGING_COLOR = False 
FILENAME = SAVE_DIR+"my.log"
EVALUATION = not TRAIN

# if train, not restore, train from scratch
SAVE_WORLD_CONFIG = True
LOAD_WORLD_CONFIG = True

SAVE_WORLD_CONFIG_DIR = SAVE_DIR + "world.config"
LOAD_WORLD_CONFIG_DIR = " "


if LOAD_WORLD_CONFIG_DIR == " ":
    LOAD_WORLD_CONFIG_DIR = SAVE_WORLD_CONFIG_DIR



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
DISCRETE_ACTION_INPUT = True
# TODO: continuous action space, the range of action based on catalog prices
# TODO: fixed catalog prices or ?
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
DIM_M = 3
DIM_B = 3
TRAINING_AGENT_TYPES = ["drl_negotiation.agents.myagent.MyConcurrentBasedAgent", "scml.scml2020.DecentralizingAgent"]
REW_FACTOR = 0.2
NEGOTIATION_SPEED = 1
QUANTITY = 10
TIME = MAX_EPISODE_LEN
UNIT_PRICE = 100
# share reward
COLLABORATIVE = False

# train update steps, based on train step during training
UPDATE_TRAINER_STEP = 100
################################################
# scml scenario
################################################

# used in SCML2020World, considered as days
N_STEPS = MAX_EPISODE_LEN

# scml scenario, concurrent
TRAINING_AGENT_TYPES_CONCURRENT = ["drl_negotiation.agents.myagent.MyConcurrentBasedAgent", "drl_negotiation.agents.myagent.MyOpponentAgent"]

################################################
# running world
################################################
#RUNNING_AGENT_TYPES = TRAINING_AGENT_TYPES
DIM_S = 2
RUNNING_AGENT_TYPES = TRAINING_AGENT_TYPES