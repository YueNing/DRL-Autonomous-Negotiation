from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.utils import make_env
import logging

# Training from scratch: set TRAIN, SAVE_WORLD_CONFIG as True
# Training from checkpoints: set TRAIN as True, SAVE_WORLD_CONFIG as False
# Evaluation from trained model: set TRAIN, SAVE_WORLD_CONFIG as False

TRAIN = False
# train from checkpoints
RESTORE = True
EVALUATION = not TRAIN

# if train, not restore, train from scratch
SAVE_WORLD_CONFIG = True

if RESTORE and TRAIN:
    SAVE_WORLD_CONFIG = False
if EVALUATION:
    SAVE_WORLD_CONFIG = False

LOAD_WORLD_CONFIG = not SAVE_WORLD_CONFIG

SAVE_DIR = "./world.config"
LOAD_DIR = " "
TRAIN_EPISODES = 10

if LOAD_DIR == " ":
    LOAD_DIR = SAVE_DIR

# make environment
env = make_env('scml', save_config=SAVE_WORLD_CONFIG, load_config=LOAD_WORLD_CONFIG, save_dir=SAVE_DIR, load_dir=LOAD_DIR)

# train model
if TRAIN:
    model = MADDPGModel(env=env, verbose=0, logging_level=logging.DEBUG, restore=RESTORE)
    model.learn(train_episodes=TRAIN_EPISODES)

# reset the environment
obs_n = env.reset()

# load model
if EVALUATION:
    # if evaluation, set the _init_setup_model as True, will initial the model and load trained parameters
    model = MADDPGModel(env=env, verbose=0, logging_level=logging.INFO, _init_setup_model=True)
    for i in range(100):
        action_n = model.predict(obs_n, train=TRAIN)
        print(f"{i}:{action_n}")
        obs_n, rew_n, done_n, info_n = env.step(action_n)
        env.render()

    env.close()
