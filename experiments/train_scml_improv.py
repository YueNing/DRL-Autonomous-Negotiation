from drl_negotiation.core.train.maddpg import MADDPGModel
from drl_negotiation.core.utils.multi_agents_utils import make_env
from drl_negotiation.core.config.hyperparameters import *
import logging

# make environment
env = make_env('scml', save_config=SAVE_WORLD_CONFIG, load_config=LOAD_WORLD_CONFIG, save_dir=SAVE_WORLD_CONFIG_DIR, load_dir=LOAD_WORLD_CONFIG_DIR)

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
