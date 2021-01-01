from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.utils import make_env
import logging

TRAIN = False
EVALUATION = not TRAIN
SAVE_WORLD_CONFIG = True
LOAD_WORLD_CONFIG = False
SAVE_DIR = "./world.config"
LOAD_DIR = " "

if LOAD_DIR == " ":
    LOAD_DIR = SAVE_DIR

# make environment
env = make_env('scml', save_config=SAVE_WORLD_CONFIG, load_config=LOAD_WORLD_CONFIG, save_dir=SAVE_DIR, load_dir=LOAD_DIR)

# train model
if TRAIN:
    model = MADDPGModel(env=env, verbose=0, logging_level=logging.DEBUG)
    model.learn(train_episodes=100)

# reset the environment
obs_n = env.reset()

# load model
if EVALUATION:
    model = MADDPGModel(env=env, verbose=0, logging_level=logging.INFO, _init_setup_model=True)

    for i in range(1000):
        action_n = model.predict(obs_n, train=False)
        obs_n, rew_n, done_n, info_n = env.step(action_n)
        env.render()

    env.close()
