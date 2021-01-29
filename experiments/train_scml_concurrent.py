from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.core.hyperparameters import *
from drl_negotiation.utils.utils import make_env
from drl_negotiation.utils.utils import logging_setup
from drl_negotiation.utils.plots import show_agent_rewards, show_ep_rewards, cumulative_reward
import logging

logging_setup(logging.INFO)

env = make_env('scml_concurrent',
               save_config=SAVE_WORLD_CONFIG,
               load_config=LOAD_WORLD_CONFIG,
               save_dir=SAVE_WORLD_CONFIG_DIR,
               load_dir=LOAD_WORLD_CONFIG_DIR,
               normalize=True
               )

model = MADDPGModel(env=env, verbose=0, restore=RESTORE)
final_ep_rewards, final_ep_extra_rewards, agent_rewards, extra_agent_rewards, episode_rewards, episode_extra_rewards, _ = model.learn(train_episodes=10)

show_ep_rewards(final_ep_rewards + final_ep_extra_rewards, model, extra=True)
show_agent_rewards(agent_rewards + extra_agent_rewards, model, extra=True)
cumulative_reward(episode_rewards + episode_extra_rewards, model, extra=True)

obs_n,_ = env.reset()
for i in range(10):
    action_n = model.predict(obs_n, train=False)
    es = env.step(action_n)
    obs_n = es.observation
    result = env.render("ascii")
    print(result)

env.close()
