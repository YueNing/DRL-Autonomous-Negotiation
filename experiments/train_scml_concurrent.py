from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.core.hyperparameters import *
from drl_negotiation.utils.utils import make_env
from drl_negotiation.utils.utils import logging_setup
from drl_negotiation.utils.plots import show_agent_rewards, show_ep_rewards, cumulative_reward
import logging

logging_setup(logging.ERROR)

env = make_env('scml_concurrent',
               save_config=SAVE_WORLD_CONFIG,
               load_config=LOAD_WORLD_CONFIG,
               save_dir=SAVE_WORLD_CONFIG_DIR,
               load_dir=LOAD_WORLD_CONFIG_DIR
               )

model = MADDPGModel(env=env, verbose=0)
final_ep_rewards, agent_rewards, episode_rewards, _ = model.learn(train_episodes=20)

show_ep_rewards(final_ep_rewards, model)
show_agent_rewards(agent_rewards, model)
cumulative_reward(episode_rewards)

obs_n = env.reset()
for i in range(10):
    action_n = model.predict(obs_n, train=False)
    obs_n, rew_n, done_n, info_n = env.step(action_n)
    env.render()

env.close()
