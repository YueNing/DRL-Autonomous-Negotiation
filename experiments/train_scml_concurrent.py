from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.hyperparameters import *
from drl_negotiation.utils import make_env
import logging

env = make_env('scml_concurrent',
               save_config=SAVE_WORLD_CONFIG,
               load_config=LOAD_WORLD_CONFIG,
               save_dir=SAVE_WORLD_CONFIG_DIR,
               load_dir=LOAD_WORLD_CONFIG_DIR
               )

model = MADDPGModel(env=env, verbose=0)
model.learn(train_episodes=100)

obs_n = env.reset()
for i in range(1000):
    action_n = model.predict(obs_n)
    obs_n, rew_n, done_n, info_n = env.step(action_n)
    env.render()

env.close()
