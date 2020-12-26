from drl_negotiation.a2c.a2c import MADDPGModel
from drl_negotiation.utils import make_env

env = make_env('scml')

model = MADDPGModel(env=env)
model.learn(train_episodes=100)

obs_n = env.reset()
for i in range(1000):
    action_n = model.predict(obs_n)
    obs_n, rew_n, done_n, info_n = env.step(action_n)
    env.render()

env.close()
